import datetime
import random
from typing import Dict, List, Optional, Tuple, TypedDict
import unittest

from schulze_condorcet import schulze_evaluate, Candidate, Vote
from schulze_condorcet.schulze_condorcet import DetailedResultLevel as DRL
from schulze_condorcet.strength import margin, winning_votes


class ClassicalTestCase(TypedDict):
    input: Dict[Optional[Tuple[Candidate, ...]], int]
    condensed: Vote
    detailed: List[DRL]


class PreferentialTestCase(TypedDict):
    input: List[Vote]
    condensed: Vote
    detailed: List[DRL]


class MyTest(unittest.TestCase):
    def test_classical_voting(self) -> None:
        bar = Candidate('0')
        c1 = Candidate('1')
        c2 = Candidate('2')
        c3 = Candidate('3')
        c4 = Candidate('4')
        c5 = Candidate('5')

        def _classical_votes(spec: Dict[Optional[Tuple[Candidate, ...]], int],
                             candidates: Tuple[Candidate, ...]) -> List[Vote]:
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
            return [Vote(v) for v in votes]

        candidates = (c1, c2, c3, c4, c5)
        candidates_with_bar = (bar,) + candidates

        test_1: ClassicalTestCase = {
            'input': {(c1,): 3, (c2,): 2, (c3,): 1, (c4,): 0, (c5,): 0, tuple(): 0, None: 0},
            'condensed': Vote("0=1>2>3>4=5"),
            'detailed': [
                DRL(preferred=[bar, c1], rejected=[c2], support=4, opposition=2),
                DRL(preferred=[c2], rejected=[c3], support=2, opposition=1),
                DRL(preferred=[c3], rejected=[c4, c5], support=1, opposition=0)
            ]
        }
        test_2: ClassicalTestCase = {
            'input': {(c1,): 9, (c2,): 0, (c3,): 2, (c4,): 1, (c5,): 8, tuple(): 1, None: 5},
            'condensed': Vote("0>1>5>3>4>2"),
            'detailed': [
                DRL(preferred=[bar], rejected=[c1], support=12, opposition=9),
                DRL(preferred=[c1], rejected=[c5], support=9, opposition=8),
                DRL(preferred=[c5], rejected=[c3], support=8, opposition=2),
                DRL(preferred=[c3], rejected=[c4], support=2, opposition=1),
                DRL(preferred=[c4], rejected=[c2], support=1, opposition=0)
            ]
        }
        test_3: ClassicalTestCase = {
            'input': {(c1,): 9, (c2,): 8, (c3,): 2, (c4,): 2, (c5,): 8, tuple(): 5, None: 5},
            'condensed': Vote("0>1>2=5>3=4"),
            'detailed': [
                DRL(preferred=[bar], rejected=[c1], support=25, opposition=9),
                DRL(preferred=[c1], rejected=[c2, c5], support=9, opposition=8),
                DRL(preferred=[c2, c5], rejected=[c3, c4], support=8, opposition=2)
            ]
        }
        test_4: ClassicalTestCase = {
            'input': {(c1, c2, c3): 2, (c1, c2): 3, (c3,): 3, (c1, c3): 1, (c2,): 1},
            'condensed': Vote("1=2=3>0>4=5"),
            'detailed': [
                DRL(preferred=[c1, c2, c3], rejected=[bar], support=6, opposition=4),
                DRL(preferred=[bar], rejected=[c4, c5], support=10, opposition=0)
            ]
        }

        for metric in {margin, winning_votes}:
            for test in [test_1, test_2, test_3, test_4]:
                with self.subTest(test=test, metric=metric):
                    condensed, detailed = schulze_evaluate(
                        _classical_votes(test['input'], candidates),
                        candidates_with_bar, strength=metric)
                    self.assertEqual(test['condensed'], condensed)
                    self.assertEqual(test['detailed'], detailed)

    def test_preferential_voting(self) -> None:
        bar = Candidate('0')
        c1 = Candidate('1')
        c2 = Candidate('2')
        c3 = Candidate('3')
        c4 = Candidate('4')

        candidates = (bar, c1, c2, c3, c4)
        # this base set is designed to have a nearly homogeneous
        # distribution (meaning all things are preferred by at most one
        # vote)
        base = [Vote("0>1>2>3>4"),
                Vote("4>3>2>1>0"),
                Vote("4=0>1=3>2"),
                Vote("3>0>2=4>1"),
                Vote("1>2=3>4=0"),
                Vote("2>1>4>0>3")]

        # the advanced set causes an even more perfect equilibrium
        advanced = [Vote("4>2>3>1=0"),
                    Vote("0>1=3>2=4"),
                    Vote("1=2>0=3=4"),
                    Vote("0=3=4>1=2")]

        tests: List[PreferentialTestCase] = [
            {
                'input': base,
                'condensed': Vote("0=1>3>2>4"),
                'detailed': []
            },
            {
                'input': base + [Vote("4>2>3>0>1")],
                'condensed': Vote("2=4>3>0>1"),
                'detailed': []
            },
            {
                'input': base + [Vote("4>2>3>1=0")],
                'condensed': Vote("2=4>1=3>0"),
                'detailed': []
            },
            {
                'input': base + [Vote("4>2>3>1=0"), Vote("0>1=3>2=4")],
                'condensed': Vote("0=3=4>1=2"),
                'detailed': []
            },
            {
                'input': base + [Vote("4>2>3>1=0"), Vote("0>1=3>2=4"), Vote("1=2>0=3=4")],
                'condensed': Vote("1=2>0=3=4"),
                'detailed': []
            },
            {
                'input': base + advanced,
                'condensed': Vote("0=3=4>1=2"),
                'detailed': []
            },
            {
                'input': base + advanced + [Vote("0>1=2=3=4")],
                'condensed': Vote("0>1=3=4>2"),
                'detailed': []
            },
            {
                'input': base + advanced + [Vote("1>0=2=3=4")],
                'condensed': Vote("0=1>3=4>2"),
                'detailed': []
            },
            {
                'input': base + advanced + [Vote("2>0=1=3=4")],
                'condensed': Vote("2=3>0=4>1"),
                'detailed': []
            },
            {
                'input': base + advanced + [Vote("3>0=1=2=4")],
                'condensed': Vote("3>0=2=4>1"),
                'detailed': []
            },
            {
                'input': base + advanced + [Vote("4>0=1=2=3")],
                'condensed': Vote("4>0=3>1=2"),
                'detailed': []
            },
            {
                'input': base + advanced + [Vote("0>3>4=1>2")],
                'condensed': Vote("0>3>1=4>2"),
                'detailed': []
            },
            {
                'input': base + advanced + [Vote("0>3>4>1>2")],
                'condensed': Vote("0>3>4>1>2"),
                'detailed': []
            },
            {
                'input': base + advanced + [Vote("2>1>4>3>0")],
                'condensed': Vote("2>1>4>3>0"),
                'detailed': []
            },
            {
                'input': base + advanced + [Vote("4>3>2>1>0")],
                'condensed': Vote("4>3>2>0=1"),
                'detailed': []
            },
            {
                'input': base + advanced + [Vote("0>1>2>3>4")],
                'condensed': Vote("0>1>2=3>4"),
                'detailed': []
            },
            {
                'input': base + advanced + [Vote("0=1=2=3>4")],
                'condensed': Vote("0=3>1=2>4"),
                'detailed': []
            },
            {
                'input': base + advanced + [Vote("0=1=2=4>3")],
                'condensed': Vote("0=2=4>1>3"),
                'detailed': []
            },
            {
                'input': base + advanced + [Vote("0=1=3=4>2")],
                'condensed': Vote("0=3=4>1>2"),
                'detailed': []
            },
            {
                'input': base + advanced + [Vote("0=2=3=4>1")],
                'condensed': Vote("0=3=4>2>1"),
                'detailed': []
            },
            {
                'input': base + advanced + [Vote("1=2=3=4>0")],
                'condensed': Vote("1=3=4>2>0"),
                'detailed': []
            },
        ]

        for metric in {margin, winning_votes}:
            for test in tests:
                with self.subTest(test=test, metric=metric):
                    condensed, detailed = schulze_evaluate(
                        test['input'], candidates, strength=metric)
                    self.assertEqual(test['condensed'], condensed)

    def test_runtime(self) -> None:
        # silly test, since I just realized, that the algorithm runtime is
        # linear in the number of votes, but a bit more scary in the number
        # of candidates
        bar = Candidate('0')
        c1 = Candidate('1')
        c2 = Candidate('2')
        c3 = Candidate('3')
        c4 = Candidate('4')

        candidates = (bar, c1, c2, c3, c4)
        votes = []
        for _ in range(2000):
            parts = list(candidates)
            random.shuffle(parts)
            relations = (random.choice(('=', '>')) for _ in range(len(candidates)))
            vote = ''.join(c + r for c, r in zip(candidates, relations))
            # zip the last char of the string
            vote = Vote(vote[:-1])
            votes.append(vote)
        times = {}
        for num in (10, 100, 1000, 2000):
            start = datetime.datetime.utcnow()
            for _ in range(10):
                schulze_evaluate(votes[:num], candidates)
            stop = datetime.datetime.utcnow()
            times[num] = stop - start
        reference = datetime.timedelta(milliseconds=5)
        for num, delta in times.items():
            self.assertGreater(num * reference, delta)

    def test_candidates_consistency(self) -> None:
        bar = Candidate('0')
        c1 = Candidate('1')
        c2 = Candidate('2')
        c3 = Candidate('3')
        einstein = Candidate('einstein')
        hawking = Candidate('hawking')
        bose = Candidate('bose')
        fermi = Candidate('fermi')

        # superfluous candidates in votes
        candidates = (bar, c1, c2, c3)
        votes = (Vote('0=1>einstein=2=3'), Vote('hawking>1=2>0=3'))
        with self.assertRaises(ValueError) as cm:
            schulze_evaluate(votes, candidates)
        self.assertEqual(str(cm.exception), "Superfluous candidate in vote string.")

        # missing candidates in votes
        candidates = (bose, einstein, hawking, fermi)
        votes = (Vote('fermi=bose>einstein'), Vote('einstein>hawking'))
        with self.assertRaises(ValueError) as cm:
            schulze_evaluate(votes, candidates)
        self.assertEqual(str(cm.exception), "Missing candidate in vote string.")

        # duplicated candidates in votes
        candidates = (einstein, fermi, bose, bar)
        votes = (Vote('einstein=einstein>bose>fermi=0'), Vote('bose>einstein=fermi=bose=0'))
        with self.assertRaises(ValueError) as cm:
            schulze_evaluate(votes, candidates)
        self.assertEqual(str(cm.exception),
                         "Every candidate must occur exactly once in each vote.")

    def test_result_order(self) -> None:
        c0 = Candidate("0")
        c1 = Candidate("1")
        c2 = Candidate("2")

        candidates = (c0, c1, c2)
        reference_votes = [Vote("0=1>2"), Vote("0=1=2")]
        reference_condensed, _ = schulze_evaluate(reference_votes, candidates)
        self.assertEqual("0=1>2", reference_condensed)

        # result is identical under arbitrary order of incoming votes
        votes = [Vote("0=1=2"), Vote("0=1>2")]
        condensed, _ = schulze_evaluate(votes, candidates)
        self.assertEqual(reference_condensed, condensed)

        # result is identical under arbitrary sorting of equal candidates in each vote
        votes = [Vote("1=0>2"), Vote("1=2=0")]
        condensed, _ = schulze_evaluate(votes, candidates)
        self.assertEqual(reference_condensed, condensed)

        # result is stable but not identical under different sorting of candidates
        candidates = (c1, c0, c2)
        condensed, _ = schulze_evaluate(reference_votes, candidates)
        self.assertEqual("1=0>2", condensed)


if __name__ == '__main__':
    unittest.main()
