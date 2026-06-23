"""Reddit ingestion via PRAW.

Scope & honesty about constraints
---------------------------------
The official Reddit API does not allow arbitrary historical range queries
(Pushshift-style). What it *does* expose reliably is listing endpoints:
`top(time_filter=...)`, `new`, `hot`, plus search. To obtain content that
naturally spans multiple time windows we pull `top(time_filter="year")` and
`new` for each subreddit and then **bucket by `created_utc`** downstream
(see TemporalProcessor). Over an active subreddit, `top`-of-year alone yields
posts distributed across many months — enough to populate ≥3 windows.

PRAW transparently handles OAuth, rate limiting (it sleeps to respect the
60 req/min budget) and pagination. We additionally cap comment fan-out
(`replace_more(limit=0)` + depth cap) so a single viral thread can't dominate
ingestion cost.
"""
from __future__ import annotations

import time
from typing import Iterator, List

from config import get_settings
from utils.logging import get_logger
from utils.models import ContentType, RedditDocument
from utils.text import clean_text

logger = get_logger("ingestion.scraper")


class RedditScraper:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._reddit = None  # lazy

    def _client(self):
        if self._reddit is not None:
            return self._reddit
        import praw  # lazy import

        s = self._settings
        if not (s.reddit_client_id and s.reddit_client_secret):
            raise RuntimeError(
                "Reddit credentials missing. Set REDDIT_CLIENT_ID / "
                "REDDIT_CLIENT_SECRET in your .env."
            )
        kwargs = dict(
            client_id=s.reddit_client_id,
            client_secret=s.reddit_client_secret,
            user_agent=s.reddit_user_agent,
        )
        if s.reddit_username and s.reddit_password:
            kwargs.update(username=s.reddit_username, password=s.reddit_password)
        self._reddit = praw.Reddit(**kwargs)
        self._reddit.read_only = True
        logger.info("PRAW client ready (read_only=%s)", self._reddit.read_only)
        return self._reddit

    # ------------------------------------------------------------------
    def scrape(
        self,
        subreddits: List[str] | None = None,
        *,
        post_limit: int | None = None,
        comment_limit: int | None = None,
    ) -> Iterator[RedditDocument]:
        """Yield posts and their comments across the configured subreddits."""
        s = self._settings
        subreddits = subreddits or s.reddit_subreddits
        post_limit = post_limit or s.scrape_post_limit
        comment_limit = comment_limit or s.scrape_comment_limit
        reddit = self._client()

        for name in subreddits:
            logger.info("Scraping r/%s (top=year + new, limit=%d)", name, post_limit)
            sub = reddit.subreddit(name)
            seen: set[str] = set()
            # Two listings broaden temporal coverage: top(year) spreads across
            # months; new captures the most recent window.
            listings = [
                sub.top(time_filter="year", limit=post_limit),
                sub.new(limit=max(post_limit // 2, 20)),
            ]
            for listing in listings:
                for submission in listing:
                    fullname = f"t3_{submission.id}"
                    if fullname in seen:
                        continue
                    seen.add(fullname)
                    yield self._post_to_doc(submission, name)
                    yield from self._comments(submission, name, comment_limit)
                    time.sleep(0.05)  # gentle pacing on top of PRAW's limiter

    # ------------------------------------------------------------------
    def _post_to_doc(self, submission, subreddit: str) -> RedditDocument:
        fullname = f"t3_{submission.id}"
        return RedditDocument(
            doc_id=fullname,
            type=ContentType.POST,
            subreddit=subreddit,
            author=str(submission.author) if submission.author else "[deleted]",
            title=clean_text(submission.title),
            body=clean_text(getattr(submission, "selftext", "") or ""),
            created_utc=float(submission.created_utc),
            score=int(submission.score),
            permalink=submission.permalink,
            parent_id=None,
            root_post_id=fullname,
            edited_utc=float(submission.edited) if submission.edited else None,
        )

    def _comments(self, submission, subreddit: str, comment_limit: int):
        try:
            submission.comments.replace_more(limit=0)  # drop "load more" stubs
        except Exception as exc:  # noqa: BLE001
            logger.warning("replace_more failed on %s: %s", submission.id, exc)
            return
        root_fullname = f"t3_{submission.id}"
        count = 0
        max_depth = self._settings.scrape_comment_depth
        for comment in submission.comments.list():
            if count >= comment_limit:
                break
            depth = getattr(comment, "depth", 0)
            if depth is not None and depth > max_depth:
                continue
            body = clean_text(getattr(comment, "body", "") or "")
            if not body or body in ("[deleted]", "[removed]"):
                continue
            count += 1
            yield RedditDocument(
                doc_id=f"t1_{comment.id}",
                type=ContentType.COMMENT,
                subreddit=subreddit,
                author=str(comment.author) if comment.author else "[deleted]",
                title="",
                body=body,
                created_utc=float(comment.created_utc),
                score=int(comment.score),
                permalink=comment.permalink,
                parent_id=comment.parent_id,           # t1_ or t3_
                root_post_id=root_fullname,
                edited_utc=float(comment.edited) if comment.edited else None,
            )
