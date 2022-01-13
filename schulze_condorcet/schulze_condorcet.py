import itertools
from gettext import gettext as _
from typing import (
    Collection, Container, List, Mapping, Tuple, Sequence
)

from schulze_condorcet.util import as_vote_string, as_vote_tuples
from schulze_condorcet.strength import winning_votes
from schulze_condorcet.types import (
    Candidate, DetailedResultLevel, LinkStrength, PairwisePreference, SchulzeResult,
    StrengthCallback, VoteString
)


def _schulze_winners(d: Mapping[Tuple[Candidate, Candidate], int],
                     candidates: Sequence[Candidate]) -> List[Candidate]:
    """This is the abstract part of the Schulze method doing the actual work.

    The candidates are the vertices of a graph and the metric (in form
    of ``d``) describes the strength of the links between the
    candidates, that is edge weights.

    We determine the strongest path from each vertex to each other
    vertex. This gives a transitive relation, which enables us thus to
    determine winners as maximal elements.
    """
    # First determine the strongest paths
    # This is a variant of the Floyd–Warshall algorithm to determine the
    # widest path.
    p = {(x, y): d[(x, y)] for x in candidates for y in candidates}
    for i in candidates:
        for j in candidates:
            if i == j:
                continue
            for k in candidates:
                if k in {i, j}:
                    continue
                p[(j, k)] = max(p[(j, k)], min(p[(j, i)], p[(i, k)]))
    # Second determine winners
    winners = []
    for i in candidates:
        if all(p[(i, j)] >= p[(j, i)] for j in candidates):
            winners.append(i)
    return winners


def _check_consistency(votes: Collection[VoteString], candidates: Sequence[Candidate]) -> None:
    """Check that the given vote strings are consistent with the provided candidates.

    This means, each vote string contains exactly the given candidates, separated by
    '>' and '=', and each candidate occurs in each vote string exactly once.
    """
    if any(">" in candidate or "=" in candidate for candidate in candidates):
        raise ValueError(_("A candidate contains a forbidden character."))
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


def _subindex(alist: Collection[Container[str]], element: str) -> int:
    """The element is in the list at which position in the big list.

    :returns: ``ret`` such that ``element in alist[ret]``
    """
    for index, sublist in enumerate(alist):
        if element in sublist:
            return index
    raise ValueError(_("Not in list."))


def _pairwise_preference(
        votes: Collection[VoteString],
        candidates: Sequence[Candidate],
) -> PairwisePreference:
    """Calculate the pairwise preference of all candidates from all given votes."""
    counts = {(x, y): 0 for x in candidates for y in candidates}
    for vote in as_vote_tuples(votes):
        for x in candidates:
            for y in candidates:
                if _subindex(vote, x) < _subindex(vote, y):
                    counts[(x, y)] += 1
    return counts


def _schulze_evaluate_routine(
        votes: Collection[VoteString],
        candidates: Sequence[Candidate],
        strength: StrengthCallback
) -> Tuple[PairwisePreference, SchulzeResult]:
    """The routine to determine the result of the schulze-condorcet method.

    This is outsourced into this helper function to avoid duplicate code or duplicate
    calculations inside the schulze_evaluate and schulze_evaluate_detailed functions.
    """
    # First we count the number of votes preferring x to y
    counts = _pairwise_preference(votes, candidates)

    # Second we calculate a numeric link strength abstracting the problem into the realm
    # of graphs with one vertex per candidate
    d: LinkStrength = {(x, y): strength(support=counts[(x, y)],
                                        opposition=counts[(y, x)],
                                        totalvotes=len(votes))
                       for x in candidates for y in candidates}

    # Third we execute the Schulze method by iteratively determining winners
    result: SchulzeResult = []
    while True:
        done = {x for level in result for x in level}
        # avoid sets to preserve ordering
        remaining = tuple(c for c in candidates if c not in done)
        if not remaining:
            break
        winners = _schulze_winners(d, remaining)
        result.append(winners)

    return counts, result


def schulze_evaluate(
        votes: Collection[VoteString],
        candidates: Sequence[Candidate],
        strength: StrengthCallback = winning_votes
) -> VoteString:
    """Use the Schulze method to cumulate preference lists (votes) into one list (vote).

    The Schulze method is described here: http://www.9mail.de/m-schulze/schulze1.pdf.
    Also the Wikipedia article is pretty nice.

    One thing to mention is, that we do not do any tie breaking.

    For a nice set of examples see the test suite.

    Note that the candidates should already be sorted meaningful. The return of this
    function is stable under arbitrary sorting of the candidates, but only identical
    if the candidates are passed in the same order. This roots in the fact that the
    result ``1=2>0`` and ``2=1>0`` carry the same meaning but are not identical.
    Therefore, we determine the order of candidates equal to each other in the final
    result by the order of those in the explicitly passed in candidates.

    The return of this function is identical under arbitrary sorting of the votes passed
    in. Moreover, the order of equal candidates in the passed in votes does not matter.

    :param votes: The vote strings on which base we want to determine the overall
      preference. One vote has the form ``3>0>1=2>4``, where the names between the
      relation signs are exactly those passed in with the ``candidates`` parameter.
    :param candidates: We require that the candidates be explicitly passed. This allows
      for more flexibility (like returning a useful result for zero votes).
    :param strength: A function which will be used as the metric on the graph of all
      candidates. See `strength.py` for more detailed information.
    :returns: A vote string, reflecting the overall preference.
    """
    # Validate votes and candidate input to be consistent
    _check_consistency(votes, candidates)

    _, result = _schulze_evaluate_routine(votes, candidates, strength)

    # Construct a vote string reflecting the overall preference
    return as_vote_string(result)


def schulze_evaluate_detailed(
        votes: Collection[VoteString],
        candidates: Sequence[Candidate],
        strength: StrengthCallback = winning_votes
) -> List[DetailedResultLevel]:
    """Construct a more detailed representation of the result by adding some stats.

    This works equally to the schulze_evaluate function but constructs a more detailed
    result, including how much of a difference there was between the individual levels
    of preference in the overall result.
    """
    # Validate votes and candidate input to be consistent
    _check_consistency(votes, candidates)

    counts, result = _schulze_evaluate_routine(votes, candidates, strength)

    # Construct the DetailedResult. This contains a list of dicts, one for each
    # level of preference, containing the preferred and rejected candidates and the
    # numbers of support and opposition (the pairwise preference) between all
    # pairwise combinations of preferred and rejected candidates.
    detailed: List[DetailedResultLevel] = list()
    for preferred_candidates, rejected_candidates in zip(result, result[1:]):
        level: DetailedResultLevel = {
            # TODO maybe use simply tuples instead of lists here?
            'preferred': list(preferred_candidates),
            'rejected': list(rejected_candidates),
            'support': {
                (preferred, rejected): counts[preferred, rejected]
                for preferred in preferred_candidates
                for rejected in rejected_candidates},
            'opposition': {
                (preferred, rejected): counts[rejected, preferred]
                for preferred in preferred_candidates
                for rejected in rejected_candidates}
        }
        detailed.append(level)
    return detailed


def pairwise_preference(
        votes: Collection[VoteString],
        candidates: Sequence[Candidate],
) -> PairwisePreference:
    """Calculate the pairwise preference of all candidates from all given votes.

    While this does not yet reveal the overall preference, it can give some more
    insights in the sentiments of the voters regarding two candidates compared to each
    other.
    """
    # Validate votes and candidate input to be consistent
    _check_consistency(votes, candidates)

    return _pairwise_preference(votes, candidates)
