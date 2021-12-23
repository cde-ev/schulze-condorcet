from typing import (
    Dict, List, NewType, Protocol, Tuple, TypedDict
)


# A single vote, represented as string. Consists only of the candidates,
# separated by '>' and '=' to express the preference between the candidates.
VoteString = NewType('VoteString', str)
# A single candidate. Can be any string, expect that it must not contain the special
# characters '>' and '='.
Candidate = NewType('Candidate', str)
# A single vote, split into separate levels accordingly to (descending) preference.
# All candidates at the same level (in the same inner tuple) have equal preference.
VoteTuple = Tuple[Tuple[Candidate, ...], ...]
# We accept VoteLists instead of VoteTuples for convenience.
VoteList = List[List[Candidate]]
# How many voters prefer the first candidate over the second candidate.
PairwisePreference = Dict[Tuple[Candidate, Candidate], int]
# The link strength between two candidates.
LinkStrength = Dict[Tuple[Candidate, Candidate], int]
# The result of the schulze_condorcet method. This has the same structure as a VoteList.
SchulzeResult = List[List[Candidate]]


class DetailedResultLevel(TypedDict):
    preferred: List[Candidate]
    rejected: List[Candidate]
    support: PairwisePreference     # between pairs of preferred and rejected candidates
    opposition: PairwisePreference  # between pairs of rejected and preferred candidates


class StrengthCallback(Protocol):
    """The interface every strength function has to implement."""
    def __call__(self, *, support: int, opposition: int, totalvotes: int) -> int: ...
