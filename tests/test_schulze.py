import datetime
import random
from typing import Dict, List, Optional, Sequence, Tuple
import unittest

from schulze_condorcet import schulze_evaluate
from schulze_condorcet.strength import margin, winning_votes


class MyTest(unittest.TestCase):
    def test_schulze_ordinary(self) -> None:
        bar = '0'

        def _ordinary_votes(spec: Dict[Optional[Tuple[str, ...]], int],
                            candidates: Tuple[str, ...]) -> List[str]:
            votes = []
            for winners, number in spec.items():
                if winners is None:
                    # abstention
                    vote = '='.join(candidates + (bar,))
                elif not winners:
                    vote = bar + '>' + '='.join(candidates)
                else:
                    vote = '='.join(winners) + '>' + bar + '>' + '='.join(
                        c for c in candidates if c not in winners)
                votes += [vote] * number
            return votes

        candidates = ('1', '2', '3', '4', '5')
        candidates_with_bar = (bar, ) + candidates
        tests: Tuple[Tuple[str, Dict[Optional[Tuple[str, ...]], int]], ...] = (
            ("0=1>2>3>4=5", {('1',): 3, ('2',): 2, ('3',): 1, ('4',): 0,
                             ('5',): 0, tuple(): 0, None: 0}),
            ("0>1>5>3>4>2", {('1',): 9, ('2',): 0, ('3',): 2, ('4',): 1,
                             ('5',): 8, tuple(): 1, None: 5}),
            ("0>1>2=5>3=4", {('1',): 9, ('2',): 8, ('3',): 2, ('4',): 2,
                             ('5',): 8, tuple(): 5, None: 5}),
            ("1=2=3>0>4=5", {('1', '2', '3'): 2, ('1', '2',): 3, ('3',): 3,
                             ('1', '3'): 1, ('2',): 1}),
        )
        for metric in {margin, winning_votes}:
            for expectation, spec in tests:
                with self.subTest(spec=spec, metric=metric):
                    condensed, _ = schulze_evaluate(_ordinary_votes(spec, candidates),
                                                    candidates_with_bar, strength=metric)
                    self.assertEqual(expectation, condensed)

    def test_schulze(self) -> None:
        candidates = ('0', '1', '2', '3', '4')
        # this base set is designed to have a nearly homogeneous
        # distribution (meaning all things are preferred by at most one
        # vote)
        base = ("0>1>2>3>4",
                "4>3>2>1>0",
                "4=0>1=3>2",
                "3>0>2=4>1",
                "1>2=3>4=0",
                "2>1>4>0>3")
        # the advanced set causes an even more perfect equilibrium
        advanced = ("4>2>3>1=0",
                    "0>1=3>2=4",
                    "1=2>0=3=4",
                    "0=3=4>1=2")
        tests: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
            ("0=1>3>2>4", tuple()),
            ("2=4>3>0>1", ("4>2>3>0>1",)),
            ("2=4>1=3>0", ("4>2>3>1=0",)),
            ("0=3=4>1=2", ("4>2>3>1=0", "0>1=3>2=4")),
            ("1=2>0=3=4", ("4>2>3>1=0", "0>1=3>2=4", "1=2>0=3=4")),
            ("0=3=4>1=2", ("4>2>3>1=0", "0>1=3>2=4", "1=2>0=3=4", "0=3=4>1=2")),
            ("0=3=4>1=2", advanced),
            ("0>1=3=4>2", advanced + ("0>1=2=3=4",)),
            ("0=1>3=4>2", advanced + ("1>0=2=3=4",)),
            ("2=3>0=4>1", advanced + ("2>0=1=3=4",)),
            ("3>0=2=4>1", advanced + ("3>0=1=2=4",)),
            ("4>0=3>1=2", advanced + ("4>0=1=2=3",)),
            ("0>3>1=4>2", advanced + ("0>3>4=1>2",)),
            ("0>3>4>1>2", advanced + ("0>3>4>1>2",)),
            ("2>1>4>3>0", advanced + ("2>1>4>3>0",)),
            ("4>3>2>0=1", advanced + ("4>3>2>1>0",)),
            ("0>1>2=3>4", advanced + ("0>1>2>3>4",)),
            ("0=3>1=2>4", advanced + ("0=1=2=3>4",)),
            ("0=2=4>1>3", advanced + ("0=1=2=4>3",)),
            ("0=3=4>1>2", advanced + ("0=1=3=4>2",)),
            ("0=3=4>2>1", advanced + ("0=2=3=4>1",)),
            ("1=3=4>2>0", advanced + ("1=2=3=4>0",)),
        )
        for metric in {margin, winning_votes}:
            for expectation, addons in tests:
                with self.subTest(addons=addons, metric=metric):
                    condensed, _ = schulze_evaluate(base + addons, candidates,
                                                    strength=metric)
                    self.assertEqual(expectation, condensed)

    def test_schulze_runtime(self) -> None:
        # silly test, since I just realized, that the algorithm runtime is
        # linear in the number of votes, but a bit more scary in the number
        # of candidates
        num_evaluation_runs = 3
        reference = datetime.timedelta(microseconds=2)

        def create_random_votes(candidates: Sequence[str], n: int) -> List[str]:
            return [''.join(c + random.choice(("=", ">")) for c in random.sample(candidates, len(candidates)))[:-1]
                    for _ in range(n)]

        for num_candidates in (3, 5, 10, 20):
            candidates = tuple(map(str, range(num_candidates)))
            for num_votes in (100, 1000, 2000):
                votes = create_random_votes(candidates, num_votes)
                # Evaluation time depends linearly on number of votes and quadratically on number of candidates.
                time_limit = num_votes * num_candidates ** 2 * reference * num_evaluation_runs
                for metric in (margin, winning_votes):
                    with self.subTest(c=num_candidates, v=num_votes, m=metric.__name__):
                        runtimes = []
                        for _ in range(num_evaluation_runs):
                            start = datetime.datetime.utcnow()
                            schulze_evaluate(votes, candidates, metric)
                            runtimes.append(datetime.datetime.utcnow() - start)
                        total_runtime = sum(runtimes, datetime.timedelta())
                        self.assertLess(total_runtime, time_limit)
                        self.assertGreater(10 * total_runtime, time_limit)

    def test_schulze_candidates(self) -> None:
        # superfluous candidates in votes
        candidates = ('0', '1', '2', '3')
        votes = ('0=1>einstein=2=3', 'hawking>1=2>0=3')
        with self.assertRaises(ValueError) as cm:
            schulze_evaluate(votes, candidates)
        self.assertEqual(str(cm.exception), "Superfluous candidate in vote string.")

        # missing candidates in votes
        candidates = ('einstein', 'hawking', 'bose', 'fermi')
        votes = ('fermi=bose>einstein', 'einstein>hawking')
        with self.assertRaises(ValueError) as cm:
            schulze_evaluate(votes, candidates)
        self.assertEqual(str(cm.exception), "Missing candidate in vote string.")

        # duplicated candidates in votes
        candidates = ('einstein', 'rose', 'bose', 'fermi')
        votes = ('einstein=rose=einstein>bose>fermi', 'rose>einstein>rose=fermi=bose')
        with self.assertRaises(ValueError) as cm:
            schulze_evaluate(votes, candidates)
        self.assertEqual(str(cm.exception),
                         "Every candidate must occur exactly once in each vote.")


if __name__ == '__main__':
    unittest.main()
