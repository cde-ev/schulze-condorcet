"""Offer convenient converting and validation functions for votes and candidates.

This especially includes marking strings as Candidate or VoteString, and converting the
string-based representations of votes (like 'a>b=c=d>e') into the level-based
representations of votes (like [[a], [b, c, d], [e]]) and vice versa.
"""

import itertools
from gettext import gettext as _
from typing import Collection, List, Sequence, Union

from schulze_condorcet.types import Candidate, VoteList, VoteString, VoteTuple


def validate_candidates(candidates: Sequence[str]) -> Sequence[Candidate]:
    """Validate a sequence of candidates.

    We respect the order of the candidates, as this is also respected during evaluation
    of the votes using the schulze method."""

    if any(">" in candidate or "=" in candidate for candidate in candidates):
        raise ValueError(_("A candidate contains a forbidden character."))
    return [Candidate(candidate) for candidate in candidates]


def validate_votes(votes: Collection[str], candidates: Sequence[str]) -> Collection[VoteString]:
    """Check that the given vote strings are consistent with the provided candidates.

    This means, each vote string contains exactly the given candidates, separated by
    '>' and '=', and each candidate occurs in each vote string exactly once.
    """
    candidates = validate_candidates(candidates)
    candidates_set = set(candidates)
    for vote in as_vote_tuples(votes):
        vote_candidates = [c for c in itertools.chain.from_iterable(vote)]
        vote_candidates_set = set(vote_candidates)
        if candidates_set != vote_candidates_set:
            if candidates_set < vote_candidates_set:
                raise ValueError(_("Superfluous candidate in vote string."))
            else:
                raise ValueError(_("Missing candidate in vote string."))
        if not len(vote_candidates) == len(vote_candidates_set):
            raise ValueError(_("Every candidate must occur exactly once in each vote."))
    return [VoteString(vote) for vote in votes]


def as_vote_string(
        value: Union[VoteList, VoteTuple]
) -> VoteString:
    """Convert a level-based representation of the vote into its string representation."""
    if isinstance(value, (list, tuple)):
        return VoteString(">".join("=".join(level) for level in value))
    else:
        raise NotImplementedError(value)


def as_vote_strings(
        values: Union[Collection[VoteList], Collection[VoteTuple]]
) -> List[VoteString]:
    """Convert each value into a string representation of the vote."""
    return [as_vote_string(value) for value in values]


def as_vote_tuple(
        value: Union[str, VoteString]
) -> VoteTuple:
    """Convert a string representation of a vote into its level-based representation."""
    return (
        tuple(
            tuple(
                Candidate(candidate) for candidate in level.split('=')
            ) for level in value.split('>')
        )
    )


def as_vote_tuples(
        values: Union[Collection[str], Collection[VoteString]]
) -> List[VoteTuple]:
    """Convert string representations of votes into their level-based representations."""
    return [as_vote_tuple(value) for value in values]
