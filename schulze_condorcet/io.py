"""Offer convenient functions to convert votes and candidates into proper types.

This especially includes marking strings as Candidate or VoteString, and converting the
string-based representations of votes (like 'a>b=c=d>e') into the level-based
representations of votes (like [[a], [b, c, d], [e]]) and vice versa.
"""

from typing import Collection, List, Sequence, Union

from schulze_condorcet.types import Candidate, VoteList, VoteString, VoteTuple


def as_candidate(value: str) -> Candidate:
    """Mark a string as candidate."""
    return Candidate(value)


def as_candidates(values: Sequence[str]) -> List[Candidate]:
    """Mark a row of strings as candidates.

    We respect the order of the candidates, as this is also respected during evaluation
    of the votes using the schulze method.
    """
    return [as_candidate(value) for value in values]


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
