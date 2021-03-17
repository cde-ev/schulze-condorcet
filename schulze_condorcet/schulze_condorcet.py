from gettext import gettext as _
import itertools
from typing import Collection, Container, Dict, List, Mapping, Tuple, Union

from schulze_condorcet.strength import StrengthCallback, winning_votes


def _schulze_winners(d: Mapping[Tuple[str, str], int],
                     candidates: Collection[str]) -> List[str]:
    """This is the abstract part of the Schulze method doing the actual work.

    The candidates are the vertices of a graph and the metric (in form
    of ``d``) describes the strength of the links between the
    candidates, that is edge weights.

    We determine the strongest path from each vertex to each other
    vertex. This gives a transitive relation, which enables us thus to
    determine winners as maximal elements.
    """
    # First determine the strongest paths
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


def schulze_evaluate(votes: Collection[str],
                     candidates: Collection[str],
                     strength: StrengthCallback = winning_votes
                     ) -> Tuple[str, List[Dict[str, Union[int, List[str]]]]]:
    """Use the Schulze method to cumulate preference list into one list.

    Votes have the form ``3>0>1=2>4`` where the shortnames between the
    relation signs are exactly those passed in the ``candidates`` parameter.

    The Schulze method is described in the pdf found in the ``related``
    folder. Also the Wikipedia article is pretty nice.

    One thing to mention is, that we do not do any tie breaking.

    For a nice set of examples see the test suite.

    :param votes: The vote strings on which base we want to determine the overall
      preference. One vote has the form ``3>0>1=2>4``.
    :param candidates: We require that the candidates be explicitly passed. This allows
      for more flexibility (like returning a useful result for zero votes).
    :param strength: A function which will be used as the metric on the graph of all
      candidates. See `strength.py` for more detailed information.
    :returns: The first Element is the aggregated result, the second is an more extended
      list, containing every level (descending) as dict with some extended information.
    """
    split_votes = tuple(
        tuple(lvl.split('=') for lvl in vote.split('>')) for vote in votes)

    # Check the candidates used in each vote string are exactly those explicitly given
    # in candidates and occur exactly once
    candidates_set = set(candidates)
    for vote in split_votes:
        vote_candidates = [c for c in itertools.chain.from_iterable(vote)]
        vote_candidates_set = set(vote_candidates)
        if candidates_set != vote_candidates_set:
            if candidates_set < vote_candidates_set:
                raise ValueError("Superfluous candidate in vote string.")
            else:
                raise ValueError("Missing candidate in vote string.")
        if not len(vote_candidates) == len(vote_candidates_set):
            raise ValueError("Every candidate must occur exactly once in each vote.")

    def _subindex(alist: Collection[Container[str]], element: str) -> int:
        """The element is in the list at which position in the big list.

        :returns: ``ret`` such that ``element in alist[ret]``
        """
        for index, sublist in enumerate(alist):
            if element in sublist:
                return index
        # This line can not be reached since we validate for well-formed votes above
        raise ValueError(_("Not in list."))  # pragma: no cover

    # First we count the number of votes preferring x to y
    counts = {(x, y): 0 for x in candidates for y in candidates}
    for vote in split_votes:
        for x in candidates:
            for y in candidates:
                if _subindex(vote, x) < _subindex(vote, y):
                    counts[(x, y)] += 1

    # Second we calculate a numeric link strength abstracting the problem into the realm
    # of graphs with one vertex per candidate
    d = {(x, y): strength(support=counts[(x, y)],
                          opposition=counts[(y, x)],
                          totalvotes=len(votes))
         for x in candidates for y in candidates}

    # Third we execute the Schulze method by iteratively determining winners
    result: List[List[str]] = []
    while True:
        done = {x for level in result for x in level}
        # avoid sets to preserve ordering
        remaining = tuple(c for c in candidates if c not in done)
        if not remaining:
            break
        winners = _schulze_winners(d, remaining)
        result.append(winners)

    # Return the aggregated preference list in the same format as the input votes are.
    condensed = ">".join("=".join(level) for level in result)
    detailed = []
    for lead, follow in zip(result, result[1:]):
        level: Dict[str, Union[List[str], int]] = {
            'winner': lead,
            'loser': follow,
            'pro_votes': counts[(lead[0], follow[0])],
            'contra_votes': counts[(follow[0], lead[0])]
        }
        detailed.append(level)

    return condensed, detailed
