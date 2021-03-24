import datetime
import random
from typing import Dict, List, Optional, Tuple
import unittest

from schulze_condorcet import schulze_evaluate, Candidate, Vote
from schulze_condorcet.strength import margin, winning_votes


class MyTest(unittest.TestCase):
    def test_schulze_ordinary(self) -> None:
        bar = Candidate('0')
        c1 = Candidate('1')
        c2 = Candidate('2')
        c3 = Candidate('3')
        c4 = Candidate('4')
        c5 = Candidate('5')

        def _ordinary_votes(spec: Dict[Optional[Tuple[Candidate, ...]], int],
                            candidates: List[Candidate]) -> List[Vote]:
            votes = []
            for winners, number in spec.items():
                if winners is None:
                    # abstention
                    vote = '='.join(candidates + [bar])
                elif not winners:
                    vote = bar + '>' + '='.join(candidates)
                else:
                    vote = '='.join(winners) + '>' + bar + '>' + '='.join(
                        c for c in candidates if c not in winners)
                votes += [vote] * number
            return [Vote(v) for v in votes]

        candidates = [c1, c2, c3, c4, c5]
        candidates_with_bar = [bar] + candidates
        tests: Tuple[Tuple[Vote, Dict[Optional[Tuple[Candidate, ...]], int]], ...] = (
            (Vote("0=1>2>3>4=5"), {(c1,): 3, (c2,): 2, (c3,): 1, (c4,): 0, (c5,): 0, tuple(): 0, None: 0}),
            (Vote("0>1>5>3>4>2"), {(c1,): 9, (c2,): 0, (c3,): 2, (c4,): 1, (c5,): 8, tuple(): 1, None: 5}),
            (Vote("0>1>2=5>3=4"), {(c1,): 9, (c2,): 8, (c3,): 2, (c4,): 2, (c5,): 8, tuple(): 5, None: 5}),
            (Vote("1=2=3>0>4=5"), {(c1, c2, c3): 2, (c1, c2): 3, (c3,): 3, (c1, c3): 1, (c2,): 1}),
        )
        for metric in {margin, winning_votes}:
            for expectation, spec in tests:
                with self.subTest(spec=spec, metric=metric):
                    condensed, _ = schulze_evaluate(_ordinary_votes(spec, candidates),
                                                    candidates_with_bar, strength=metric)
                    self.assertEqual(expectation, condensed)

    def test_schulze(self) -> None:
        bar = Candidate('0')
        c1 = Candidate('1')
        c2 = Candidate('2')
        c3 = Candidate('3')
        c4 = Candidate('4')

        candidates = (bar, c1, c2, c3, c4)
        # this base set is designed to have a nearly homogeneous
        # distribution (meaning all things are preferred by at most one
        # vote)
        base = (Vote("0>1>2>3>4"),
                Vote("4>3>2>1>0"),
                Vote("4=0>1=3>2"),
                Vote("3>0>2=4>1"),
                Vote("1>2=3>4=0"),
                Vote("2>1>4>0>3"))
        # the advanced set causes an even more perfect equilibrium
        advanced = (Vote("4>2>3>1=0"),
                    Vote("0>1=3>2=4"),
                    Vote("1=2>0=3=4"),
                    Vote("0=3=4>1=2"))
        tests: Tuple[Tuple[Vote, Tuple[Vote, ...]], ...] = (
            (Vote("0=1>3>2>4"), tuple()),
            (Vote("2=4>3>0>1"), (Vote("4>2>3>0>1"),)),
            (Vote("2=4>1=3>0"), (Vote("4>2>3>1=0"),)),
            (Vote("0=3=4>1=2"), (Vote("4>2>3>1=0"), Vote("0>1=3>2=4"))),
            (Vote("1=2>0=3=4"), (Vote("4>2>3>1=0"), Vote("0>1=3>2=4"), Vote("1=2>0=3=4"))),
            (Vote("0=3=4>1=2"), (Vote("4>2>3>1=0"), Vote("0>1=3>2=4"), Vote("1=2>0=3=4"), Vote("0=3=4>1=2"))),
            (Vote("0=3=4>1=2"), advanced),
            (Vote("0>1=3=4>2"), advanced + (Vote("0>1=2=3=4"),)),
            (Vote("0=1>3=4>2"), advanced + (Vote("1>0=2=3=4"),)),
            (Vote("2=3>0=4>1"), advanced + (Vote("2>0=1=3=4"),)),
            (Vote("3>0=2=4>1"), advanced + (Vote("3>0=1=2=4"),)),
            (Vote("4>0=3>1=2"), advanced + (Vote("4>0=1=2=3"),)),
            (Vote("0>3>1=4>2"), advanced + (Vote("0>3>4=1>2"),)),
            (Vote("0>3>4>1>2"), advanced + (Vote("0>3>4>1>2"),)),
            (Vote("2>1>4>3>0"), advanced + (Vote("2>1>4>3>0"),)),
            (Vote("4>3>2>0=1"), advanced + (Vote("4>3>2>1>0"),)),
            (Vote("0>1>2=3>4"), advanced + (Vote("0>1>2>3>4"),)),
            (Vote("0=3>1=2>4"), advanced + (Vote("0=1=2=3>4"),)),
            (Vote("0=2=4>1>3"), advanced + (Vote("0=1=2=4>3"),)),
            (Vote("0=3=4>1>2"), advanced + (Vote("0=1=3=4>2"),)),
            (Vote("0=3=4>2>1"), advanced + (Vote("0=2=3=4>1"),)),
            (Vote("1=3=4>2>0"), advanced + (Vote("1=2=3=4>0"),)),
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

    def test_schulze_candidates(self) -> None:
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


if __name__ == '__main__':
    unittest.main()
