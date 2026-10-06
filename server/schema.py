"""The single response contract between the backend and the iOS app.

Every field may be empty ("" / [] / None). Empty means "not present in this reel"; the app
renders a section only when it has content. The LLM is never forced to fill anything.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from reelscout.context import CONTEXT_SCHEMA


class Location(BaseModel):
    name: str = ""
    area: str = ""  # neighbourhood / locality, e.g. "Gachibowli"
    city: str = ""
    country: str = ""
    lat: float | None = None
    lng: float | None = None
    map_url: str = ""
    confidence: float = 0.0
    precision: str = ""  # exact | area | city: how specific the pin is
    note: str = ""  # e.g. "Exact address not shared in the reel; near Manu Temple"

    @property
    def present(self) -> bool:
        return bool(self.name.strip())


class Product(BaseModel):
    name: str = ""
    brand: str = ""
    url: str = ""


class Person(BaseModel):
    name: str = ""
    role: str = ""


class Link(BaseModel):
    label: str = ""
    url: str = ""
    kind: str = ""  # github_repo | website | tool | package | app | other
    note: str = ""  # one line on what it is, e.g. "Vectorless RAG engine using tree indexes"


class Source(BaseModel):
    title: str = ""
    url: str = ""


class Event(BaseModel):
    """A dated thing found in a screenshot (event, ticket, booking, deadline). Times are ISO 8601 with offset."""

    id: str = ""
    title: str = ""
    start: str = ""
    end: str = ""
    all_day: bool = False
    timezone: str = ""  # IANA name the times were read in
    location: str = ""
    notes: str = ""
    confidence: float = 0.0
    needs_check: bool = False  # date was inferred or the time is missing: ask the user to confirm


class KeyMoment(BaseModel):
    id: str = ""
    timestamp: str = ""
    seconds: float = 0.0
    label: str = ""
    note: str = ""
    index: int = 0
    frame_url: str = ""


class Point(BaseModel):
    text: str = ""
    done: bool = False


class ReelContext(BaseModel):
    """What the LLM may decide. Everything else in the payload is deterministic."""

    title: str = ""
    summary: str = ""
    category: str = ""  # Food | Travel | Fitness | Home | Tech | Shopping | Learning | Entertainment | ""
    location: Location = Field(default_factory=Location)  # the first place, kept for older clients
    places: list[Location] = Field(default_factory=list)  # every place the reel is about, in order
    products: list[Product] = Field(default_factory=list)
    people: list[Person] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)
    things_mentioned: list[str] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list)
    useful_context: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    username: str = ""


class ReelPayload(ReelContext):
    """Full wire object returned by /api/v1/reels."""

    schema_version: int = CONTEXT_SCHEMA
    kind: str = "reel"  # reel | post | screenshot
    shot_type: str = ""  # screenshots only: event, ticket, booking, schedule, message, article, ...
    events: list[Event] = Field(default_factory=list)
    id: str
    created: str | float | None = None
    author: str = ""
    duration: str = "0:00"
    cover_url: str = ""
    status: str = "ready"
    progress: float = 1.0
    group: str = ""
    platform: str = "unknown"
    source_url: str = ""
    frame_count: int = 0
    key_moments: list[KeyMoment] = Field(default_factory=list)
    points: list[Point] = Field(default_factory=list)
    # Live processing only: per-step detail text ("0:42 video"), finding chips, failure reason.
    details: dict[str, str] = Field(default_factory=dict)
    chips: list[str] = Field(default_factory=list)
    error: str = ""
