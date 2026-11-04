"""What a reader's reply to a digest says about its articles."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rating:
    """The reader's verdict on one article: 1 liked, -1 disliked, 0 just a comment."""
    rating: int
    comment: str = ""

    @property
    def label(self) -> str:
        return {1: "👍 liked", -1: "👎 disliked", 0: "💬 noted"}[self.rating]


@dataclass
class Interpretation:
    """What a reply says: ratings by item number, and comments not about one article."""
    ratings: dict[int, Rating] = field(default_factory=dict)
    general: list[str] = field(default_factory=list)

    def filled_in_by(self, other: "Interpretation") -> "Interpretation":
        """These ratings, plus `other`'s for items these leave out or only comment on;
        `other`'s general comments."""
        ratings = dict(self.ratings)
        for item_number, rating in other.ratings.items():
            if item_number not in ratings or ratings[item_number].rating == 0:
                ratings[item_number] = rating
        return Interpretation(ratings, other.general)
