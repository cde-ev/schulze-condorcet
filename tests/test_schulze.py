import datetime
import random
from typing import Dict, List, Optional, Tuple, TypedDict
import unittest

from schulze_condorcet import schulze_evaluate, schulze_evaluate_detailed
import schulze_condorcet.util as util
from schulze_condorcet.strength import margin, winning_votes
from schulze_condorcet.types import Candidate, DetailedResultLevel as DRL, VoteString


class ClassicalTestCase(TypedDict):
    input: Dict[Optional[Tuple[Candidate, ...]], int]
    condensed: VoteString
    detailed: List[DRL]


class PreferentialTestCase(TypedDict):
    input: List[VoteString]
    condensed: VoteString
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
                             candidates: Tuple[Candidate, ...]) -> List[VoteString]:
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
            return [VoteString(v) for v in votes]

        candidates = (c1, c2, c3, c4, c5)
        candidates_with_bar = (bar,) + candidates

        test_1: ClassicalTestCase = {
            'input': {(c1,): 3, (c2,): 2, (c3,): 1, (c4,): 0, (c5,): 0, tuple(): 0, None: 0},
            'condensed': VoteString("0=1>2>3>4=5"),
            'detailed': [
                DRL(preferred=[bar, c1], rejected=[c2],
                    support={(bar, c2): 4, (c1, c2): 3},
                    opposition={(bar, c2): 2, (c1, c2): 2}),
                DRL(preferred=[c2], rejected=[c3],
                    support={(c2, c3): 2},
                    opposition={(c2, c3): 1}),
                DRL(preferred=[c3], rejected=[c4, c5],
                    support={(c3, c4): 1, (c3, c5): 1},
                    opposition={(c3, c4): 0, (c3, c5): 0})
            ]
        }
        test_2: ClassicalTestCase = {
            'input': {(c1,): 9, (c2,): 0, (c3,): 2, (c4,): 1, (c5,): 8, tuple(): 1, None: 5},
            'condensed': VoteString("0>1>5>3>4>2"),
            'detailed': [
                DRL(preferred=[bar], rejected=[c1],
                    support={(bar, c1): 12},
                    opposition={(bar, c1): 9}),
                DRL(preferred=[c1], rejected=[c5],
                    support={(c1, c5): 9},
                    opposition={(c1, c5): 8}),
                DRL(preferred=[c5], rejected=[c3],
                    support={(c5, c3): 8},
                    opposition={(c5, c3): 2}),
                DRL(preferred=[c3], rejected=[c4],
                    support={(c3, c4): 2},
                    opposition={(c3, c4): 1}),
                DRL(preferred=[c4], rejected=[c2],
                    support={(c4, c2): 1},
                    opposition={(c4, c2): 0})
            ]
        }
        test_3: ClassicalTestCase = {
            'input': {(c1,): 9, (c2,): 8, (c3,): 2, (c4,): 2, (c5,): 8, tuple(): 5, None: 5},
            'condensed': VoteString("0>1>2=5>3=4"),
            'detailed': [
                DRL(preferred=[bar], rejected=[c1],
                    support={(bar, c1): 25},
                    opposition={(bar, c1): 9}),
                DRL(preferred=[c1], rejected=[c2, c5],
                    support={(c1, c2): 9, (c1, c5): 9},
                    opposition={(c1, c2): 8, (c1, c5): 8}),
                DRL(preferred=[c2, c5], rejected=[c3, c4],
                    support={(c2, c3): 8, (c2, c4): 8, (c5, c3): 8, (c5, c4): 8},
                    opposition={(c2, c3): 2, (c2, c4): 2, (c5, c3): 2, (c5, c4): 2})
            ]
        }
        test_4: ClassicalTestCase = {
            'input': {(c1, c2, c3): 2, (c1, c2): 3, (c3,): 3, (c1, c3): 1, (c2,): 1},
            'condensed': VoteString("1=2=3>0>4=5"),
            'detailed': [
                DRL(preferred=[c1, c2, c3], rejected=[bar],
                    support={(c1, bar): 6, (c2, bar): 6, (c3, bar): 6},
                    opposition={(c1, bar): 4, (c2, bar): 4, (c3, bar): 4}),
                DRL(preferred=[bar], rejected=[c4, c5],
                    support={(bar, c4): 10, (bar, c5): 10},
                    opposition={(bar, c4): 0, (bar, c5): 0})
            ]
        }

        for metric in {margin, winning_votes}:
            for test in [test_1, test_2, test_3, test_4]:
                with self.subTest(test=test, metric=metric):
                    votes = _classical_votes(test['input'], candidates)
                    condensed = schulze_evaluate(
                        votes, candidates_with_bar, strength=metric)
                    detailed = schulze_evaluate_detailed(
                        votes, candidates_with_bar, strength=metric)
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
        base = [VoteString("0>1>2>3>4"),
                VoteString("4>3>2>1>0"),
                VoteString("4=0>1=3>2"),
                VoteString("3>0>2=4>1"),
                VoteString("1>2=3>4=0"),
                VoteString("2>1>4>0>3")]

        # the advanced set causes an even more perfect equilibrium
        advanced = [VoteString("4>2>3>1=0"),
                    VoteString("0>1=3>2=4"),
                    VoteString("1=2>0=3=4"),
                    VoteString("0=3=4>1=2")]

        tests: List[PreferentialTestCase] = [
            {
                'input': base,
                'condensed': VoteString("0=1>3>2>4"),
                'detailed': [
                    DRL(preferred=[bar, c1], rejected=[c3],
                        support={(bar, c3): 3, (c1, c3): 3},
                        opposition={(bar, c3): 3, (c1, c3): 2}),
                    DRL(preferred=[c3], rejected=[c2],
                        support={(c3, c2): 3},
                        opposition={(c3, c2): 2}),
                    DRL(preferred=[c2], rejected=[c4],
                        support={(c2, c4): 3},
                        opposition={(c2, c4): 2})
                ]
            },
            {
                'input': base + [VoteString("4>2>3>0>1")],
                'condensed': VoteString("2=4>3>0>1"),
                'detailed': [
                    DRL(preferred=[c2, c4], rejected=[c3],
                        support={(c2, c3): 3, (c4, c3): 4},
                        opposition={(c2, c3): 3, (c4, c3): 3}),
                    DRL(preferred=[c3], rejected=[bar],
                        support={(c3, bar): 4},
                        opposition={(c3, bar): 3}),
                    DRL(preferred=[bar], rejected=[c1],
                        support={(bar, c1): 4},
                        opposition={(bar, c1): 3})
                ]
            },
            {
                'input': base + [VoteString("4>2>3>1=0")],
                'condensed': VoteString("2=4>1=3>0"),
                'detailed': [
                    DRL(preferred=[c2, c4], rejected=[c1, c3],
                        support={(c2, c1): 4, (c2, c3): 3, (c4, c1): 4, (c4, c3): 4},
                        opposition={(c2, c1): 3, (c2, c3): 3, (c4, c1): 3, (c4, c3): 3}),
                    DRL(preferred=[c1, c3], rejected=[bar],
                        support={(c1, bar): 3, (c3, bar): 4},
                        opposition={(c1, bar): 3, (c3, bar): 3})
                ]
            },
            {
                'input': base + [VoteString("4>2>3>1=0"), VoteString("0>1=3>2=4")],
                'condensed': VoteString("0=3=4>1=2"),
                'detailed': [
                    DRL(preferred=[bar, c3, c4], rejected=[c1, c2],
                        support={(bar, c1): 4, (bar, c2): 4, (c3, c1): 3, (c3, c2): 4, (c4, c1): 4, (c4, c2): 3},
                        opposition={(bar, c1): 3, (bar, c2): 4, (c3, c1): 3, (c3, c2): 3, (c4, c1): 4, (c4, c2): 3})
                ]
            },
            {
                'input': base + [VoteString("4>2>3>1=0"), VoteString("0>1=3>2=4"), VoteString("1=2>0=3=4")],
                'condensed': VoteString("1=2>0=3=4"),
                'detailed': [
                    DRL(preferred=[c1, c2], rejected=[bar, c3, c4],
                        support={(c1, bar): 4, (c1, c3): 4, (c1, c4): 5, (c2, bar): 5, (c2, c3): 4, (c2, c4): 4},
                        opposition={(c1, bar): 4, (c1, c3): 3, (c1, c4): 4, (c2, bar): 4, (c2, c3): 4, (c2, c4): 3})
                ]
            },
            {
                'input': base + advanced,
                'condensed': VoteString("0=3=4>1=2"),
                'detailed': [
                    DRL(preferred=[bar, c3, c4], rejected=[c1, c2],
                        support={(bar, c1): 5, (bar, c2): 5, (c3, c1): 4, (c3, c2): 5, (c4, c1): 5, (c4, c2): 4},
                        opposition={(bar, c1): 4, (bar, c2): 5, (c3, c1): 4, (c3, c2): 4, (c4, c1): 5, (c4, c2): 4})
                ]
            },
            {
                'input': base + advanced + [VoteString("0>1=2=3=4")],
                'condensed': VoteString("0>1=3=4>2"),
                'detailed': [
                    DRL(preferred=[bar], rejected=[c1, c3, c4],
                        support={(bar, c1): 6, (bar, c3): 5, (bar, c4): 4},
                        opposition={(bar, c1): 4, (bar, c3): 4, (bar, c4): 3}),
                    DRL(preferred=[c1, c3, c4], rejected=[c2],
                        support={(c1, c2): 4, (c3, c2): 5, (c4, c2): 4},
                        opposition={(c1, c2): 4, (c3, c2): 4, (c4, c2): 4})
                ]
            },
            {
                'input': base + advanced + [VoteString("1>0=2=3=4")],
                'condensed': VoteString("0=1>3=4>2"),
                'detailed': [
                    DRL(preferred=[bar, c1], rejected=[c3, c4],
                        support={(bar, c3): 4, (bar, c4): 3, (c1, c3): 5, (c1, c4): 6},
                        opposition={(bar, c3): 4, (bar, c4): 3, (c1, c3): 4, (c1, c4): 5}),
                    DRL(preferred=[c3, c4], rejected=[c2],
                        support={(c3, c2): 5, (c4, c2): 4},
                        opposition={(c3, c2): 4, (c4, c2): 4})
                ]
            },
            {
                'input': base + advanced + [VoteString("2>0=1=3=4")],
                'condensed': VoteString("2=3>0=4>1"),
                'detailed': [
                    DRL(preferred=[c2, c3], rejected=[bar, c4],
                        support={(c2, bar): 6, (c2, c4): 5, (c3, bar): 4, (c3, c4): 4},
                        opposition={(c2, bar): 5, (c2, c4): 4, (c3, bar): 4, (c3, c4): 4}),
                    DRL(preferred=[bar, c4], rejected=[c1],
                        support={(bar, c1): 5, (c4, c1): 5},
                        opposition={(bar, c1): 4, (c4, c1): 5})
                ]
            },
            {
                'input': base + advanced + [VoteString("3>0=1=2=4")],
                'condensed': VoteString("3>0=2=4>1"),
                'detailed': [
                    DRL(preferred=[c3], rejected=[bar, c2, c4],
                        support={(c3, bar): 5, (c3, c2): 6, (c3, c4): 5},
                        opposition={(c3, bar): 4, (c3, c2): 4, (c3, c4): 4}),
                    DRL(preferred=[bar, c2, c4], rejected=[c1],
                        support={(bar, c1): 5, (c2, c1): 4, (c4, c1): 5},
                        opposition={(bar, c1): 4, (c2, c1): 4, (c4, c1): 5})
                ]
            },
            {
                'input': base + advanced + [VoteString("4>0=1=2=3")],
                'condensed': VoteString("4>0=3>1=2"),
                'detailed': [
                    DRL(preferred=[c4], rejected=[bar, c3],
                        support={(c4, bar): 4, (c4, c3): 5},
                        opposition={(c4, bar): 3, (c4, c3): 4}),
                    DRL(preferred=[bar, c3], rejected=[c1, c2],
                        support={(bar, c1): 5, (bar, c2): 5, (c3, c1): 4, (c3, c2): 5},
                        opposition={(bar, c1): 4, (bar, c2): 5, (c3, c1): 4, (c3, c2): 4})
                ]
            },
            {
                'input': base + advanced + [VoteString("0>3>4=1>2")],
                'condensed': VoteString("0>3>1=4>2"),
                'detailed': [
                    DRL(preferred=[bar], rejected=[c3],
                        support={(bar, c3): 5},
                        opposition={(bar, c3): 4}),
                    DRL(preferred=[c3], rejected=[c1, c4],
                        support={(c3, c1): 5, (c3, c4): 5},
                        opposition={(c3, c1): 4, (c3, c4): 4}),
                    DRL(preferred=[c1, c4], rejected=[c2],
                        support={(c1, c2): 5, (c4, c2): 5},
                        opposition={(c1, c2): 4, (c4, c2): 4})
                ]
            },
            {
                'input': base + advanced + [VoteString("0>3>4>1>2")],
                'condensed': VoteString("0>3>4>1>2"),
                'detailed': [
                    DRL(preferred=[bar], rejected=[c3],
                        support={(bar, c3): 5},
                        opposition={(bar, c3): 4}),
                    DRL(preferred=[c3], rejected=[c4],
                        support={(c3, c4): 5},
                        opposition={(c3, c4): 4}),
                    DRL(preferred=[c4], rejected=[c1],
                        support={(c4, c1): 6},
                        opposition={(c4, c1): 5}),
                    DRL(preferred=[c1], rejected=[c2],
                        support={(c1, c2): 5},
                        opposition={(c1, c2): 4})
                ]
            },
            {
                'input': base + advanced + [VoteString("2>1>4>3>0")],
                'condensed': VoteString("2>1>4>3>0"),
                'detailed': [
                    DRL(preferred=[c2], rejected=[c1],
                        support= {(c2, c1): 5},
                        opposition={(c2, c1): 4}),
                    DRL(preferred=[c1], rejected=[c4],
                        support={(c1, c4): 6},
                        opposition={(c1, c4): 5}),
                    DRL(preferred=[c4], rejected=[c3],
                        support={(c4, c3): 5},
                        opposition={(c4, c3): 4}),
                    DRL(preferred=[c3], rejected=[bar],
                        support={(c3, bar): 5},
                        opposition={(c3, bar): 4})
                ]
            },
            {
                'input': base + advanced + [VoteString("4>3>2>1>0")],
                'condensed': VoteString("4>3>2>0=1"),
                'detailed': [
                    DRL(preferred=[c4], rejected=[c3],
                        support={(c4, c3): 5},
                        opposition={(c4, c3): 4}),
                    DRL(preferred=[c3], rejected=[c2],
                        support={(c3, c2): 6},
                        opposition={(c3, c2): 4}),
                    DRL(preferred=[c2], rejected=[bar, c1],
                        support={(c2, bar): 6, (c2, c1): 5},
                        opposition={(c2, bar): 5, (c2, c1): 4})
                ]
            },
            {
                'input': base + advanced + [VoteString("0>1>2>3>4")],
                'condensed': VoteString("0>1>2=3>4"),
                'detailed': [
                    DRL(preferred=[bar], rejected=[c1],
                        support={(bar, c1): 6},
                        opposition={(bar, c1): 4}),
                    DRL(preferred=[c1], rejected=[c2, c3],
                        support={(c1, c2): 5, (c1, c3): 5},
                        opposition={(c1, c2): 4, (c1, c3): 4}),
                    DRL(preferred=[c2, c3], rejected=[c4],
                        support={(c2, c4): 5, (c3, c4): 5},
                        opposition={(c2, c4): 4, (c3, c4): 4})
                ]
            },
            {
                'input': base + advanced + [VoteString("0=1=2=3>4")],
                'condensed': VoteString("0=3>1=2>4"),
                'detailed': [
                    DRL(preferred=[bar, c3], rejected=[c1, c2],
                        support={(bar, c1): 5, (bar, c2): 5, (c3, c1): 4, (c3, c2): 5},
                        opposition={(bar, c1): 4, (bar, c2): 5, (c3, c1): 4, (c3, c2): 4}),
                    DRL(preferred=[c1, c2], rejected=[c4],
                        support={(c1, c4): 6, (c2, c4): 5},
                        opposition={(c1, c4): 5, (c2, c4): 4})
                ]
            },
            {
                'input': base + advanced + [VoteString("0=1=2=4>3")],
                'condensed': VoteString("0=2=4>1>3"),
                'detailed': [
                    DRL(preferred=[bar, c2, c4], rejected=[c1],
                        support={(bar, c1): 5, (c2, c1): 4, (c4, c1): 5},
                        opposition={(bar, c1): 4, (c2, c1): 4, (c4, c1): 5}),
                    DRL(preferred=[c1], rejected=[c3],
                        support={(c1, c3): 5},
                        opposition={(c1, c3): 4})
                ]
            },
            {
                'input': base + advanced + [VoteString("0=1=3=4>2")],
                'condensed': VoteString("0=3=4>1>2"),
                'detailed': [
                    DRL(preferred=[bar, c3, c4], rejected=[c1],
                        support={(bar, c1): 5, (c3, c1): 4, (c4, c1): 5},
                        opposition={(bar, c1): 4, (c3, c1): 4, (c4, c1): 5}),
                    DRL(preferred=[c1], rejected=[c2],
                        support={(c1, c2): 5},
                        opposition={(c1, c2): 4})
                ]
            },
            {
                'input': base + advanced + [VoteString("0=2=3=4>1")],
                'condensed': VoteString("0=3=4>2>1"),
                'detailed': [
                    DRL(preferred=[bar, c3, c4], rejected=[c2],
                        support={(bar, c2): 5, (c3, c2): 5, (c4, c2): 4},
                        opposition={(bar, c2): 5, (c3, c2): 4, (c4, c2): 4}),
                    DRL(preferred=[c2], rejected=[c1],
                        support={(c2, c1): 5},
                        opposition={(c2, c1): 4})
                ]
            },
            {
                'input': base + advanced + [VoteString("1=2=3=4>0")],
                'condensed': VoteString("1=3=4>2>0"),
                'detailed': [
                    DRL(preferred=[c1, c3, c4], rejected=[c2],
                        support={(c1, c2): 4, (c3, c2): 5, (c4, c2): 4},
                        opposition={(c1, c2): 4, (c3, c2): 4, (c4, c2): 4}),
                    DRL(preferred=[c2], rejected=[bar],
                        support={(c2, bar): 6},
                        opposition={(c2, bar): 5})
                ]
            },
        ]

        for metric in {margin, winning_votes}:
            for test in tests:
                with self.subTest(test=test, metric=metric):
                    condensed = schulze_evaluate(
                        test['input'], candidates, strength=metric)
                    detailed = schulze_evaluate_detailed(
                        test['input'], candidates, strength=metric)
                    self.assertEqual(test['condensed'], condensed)
                    self.assertEqual(test['detailed'], detailed)

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
            vote = VoteString(vote[:-1])
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
        votes = (VoteString('0=1>einstein=2=3'), VoteString('hawking>1=2>0=3'))
        with self.assertRaises(ValueError) as cm:
            schulze_evaluate(votes, candidates)
        self.assertEqual(str(cm.exception), "Superfluous candidate in vote string.")

        # missing candidates in votes
        candidates = (bose, einstein, hawking, fermi)
        votes = (VoteString('fermi=bose>einstein'), VoteString('einstein>hawking'))
        with self.assertRaises(ValueError) as cm:
            schulze_evaluate(votes, candidates)
        self.assertEqual(str(cm.exception), "Missing candidate in vote string.")

        # duplicated candidates in votes
        candidates = (einstein, fermi, bose, bar)
        votes = (VoteString('einstein=einstein>bose>fermi=0'), VoteString('bose>einstein=fermi=bose=0'))
        with self.assertRaises(ValueError) as cm:
            schulze_evaluate(votes, candidates)
        self.assertEqual(str(cm.exception),
                         "Every candidate must occur exactly once in each vote.")

    def test_result_order(self) -> None:
        c0 = Candidate("0")
        c1 = Candidate("1")
        c2 = Candidate("2")

        candidates = (c0, c1, c2)
        reference_votes = [VoteString("0=1>2"), VoteString("0=1=2")]
        reference_condensed = schulze_evaluate(reference_votes, candidates)
        self.assertEqual("0=1>2", reference_condensed)

        # result is identical under arbitrary order of incoming votes
        votes = [VoteString("0=1=2"), VoteString("0=1>2")]
        condensed = schulze_evaluate(votes, candidates)
        self.assertEqual(reference_condensed, condensed)

        # result is identical under arbitrary sorting of equal candidates in each vote
        votes = [VoteString("1=0>2"), VoteString("1=2=0")]
        condensed = schulze_evaluate(votes, candidates)
        self.assertEqual(reference_condensed, condensed)

        # result is stable but not identical under different sorting of candidates
        candidates = (c1, c0, c2)
        condensed = schulze_evaluate(reference_votes, candidates)
        self.assertEqual("1=0>2", condensed)

    def test_util(self) -> None:
        candidates = ["1", "2", "3"]
        # This does only static type conversion
        self.assertEqual(util.validate_candidates(candidates), candidates)

        # build the same votes, represented as string, list and tuple
        vote_str_1 = "1"
        vote_str_2 = "1=2>3"
        vote_list_1 = [[Candidate("1")]]
        vote_list_2 = [[Candidate("1"), Candidate("2")], [Candidate("3")]]
        vote_tuple_1 = ((Candidate("1"),),)
        vote_tuple_2 = ((Candidate("1"), Candidate("2")), (Candidate("3"),))
        vote_str_list = [vote_str_1, vote_str_2]
        vote_list_list = [vote_list_1, vote_list_2]
        vote_tuple_list = [vote_tuple_1, vote_tuple_2]

        self.assertEqual(vote_str_list, util.as_vote_strings(vote_list_list))
        self.assertEqual(vote_str_list, util.as_vote_strings(vote_tuple_list))

        # mixing different representations of votes is confusing and therefore not
        # recommended and forbidden by static type checking.
        # However, we ensure the outcome is right nonetheless.
        self.assertEqual(
            2*vote_str_list,
            util.as_vote_strings([*vote_list_list, *vote_tuple_list])  # type:ignore
        )

        self.assertEqual(vote_tuple_list, util.as_vote_tuples(vote_str_list))


if __name__ == '__main__':
    unittest.main()
