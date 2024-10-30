import unittest

from src.algo.boulderdash import BoulderDash


class TestBoulderDash(unittest.TestCase):

    def assert_map(self, bd: BoulderDash, map: str):
        self.assertEqual(
            BoulderDash.from_string_map(map).to_string(),
            bd.to_string(),
            f"\nExpected:\n{BoulderDash.from_string_map(map).to_string()}\nGot:\n{bd.to_string()}"
        )

    def assert_step_sequence(self, bd: BoulderDash, *maps: str):
        for map in maps:
            bd.step()
            self.assert_map(bd, map)

    def test_100_construct(self):
        bd = BoulderDash.from_string_map("""
        WWWWWWWWWW
        WDS   R  W
        WW    WWWW 
        W  P     W
        WWWWWWWWWW
        """)
        self.assertEqual(
            "WWWWWWWWWW\nWDS...R..W\nWW....WWWW\nW..P.....W\nWWWWWWWWWW\n",
            bd.to_string()
        )

    def test_200_actions(self):
        bd = BoulderDash.from_string_map("""
        WWWWWWWWWW
        W D P    W
        WWWWWWWWWW
        """)
        self.assertEqual(
            bd.RESULTS.Blocked,
            bd.apply_action(bd.ACTIONS.Up),
        )
        self.assertEqual(
            bd.RESULTS.Blocked,
            bd.apply_action(bd.ACTIONS.Down),
        )
        self.assertEqual(
            bd.RESULTS.Moved,
            bd.apply_action(bd.ACTIONS.Left),
        )
        self.assert_map(bd, """
        WWWWWWWWWW
        W DP     W
        WWWWWWWWWW
        """)
        self.assertEqual(
            bd.RESULTS.CollectedAllDiamonds,
            bd.apply_action(bd.ACTIONS.Left),
        )
        self.assert_map(bd, """
        WWWWWWWWWW
        W P      W
        WWWWWWWWWW
        """)

    def test_201_actions(self):
        bd = BoulderDash.from_string_map("""
        ..
        .P
        """)
        self.assertEqual(bd.RESULTS.Moved, bd.apply_action(bd.ACTIONS.Up))
        self.assert_map(bd, """
        .P
        ..
        """)
        self.assertEqual(bd.RESULTS.Moved, bd.apply_action(bd.ACTIONS.Left))
        self.assert_map(bd, """
        P.
        ..
        """)
        self.assertEqual(bd.RESULTS.Moved, bd.apply_action(bd.ACTIONS.Down))
        self.assertEqual(bd.RESULTS.Moved, bd.apply_action(bd.ACTIONS.Right))
        self.assert_map(bd, """
        ..
        .P
        """)

    def test_202_actions(self):
        bd = BoulderDash.from_string_map("""
        WWWWWWWWWW
        W   PRR  W
        WWWWWWWWWW
        """)
        self.assertEqual(
            bd.RESULTS.PushedRock,
            bd.apply_action(bd.ACTIONS.Right),
        )
        self.assert_map(bd, """
        WWWWWWWWWW
        W    PRR W
        WWWWWWWWWW
        """)
        self.assertEqual(
            bd.RESULTS.PushedRock,
            bd.apply_action(bd.ACTIONS.Right),
        )
        self.assert_map(bd, """
        WWWWWWWWWW
        W     PRRW
        WWWWWWWWWW
        """)
        self.assertEqual(
            bd.RESULTS.Blocked,
            bd.apply_action(bd.ACTIONS.Right),
        )

    def test_300_step(self):
        bd = BoulderDash.from_string_map("""
        WWWWWWWWWWW
        W R  R  R W
        W    W WW W
        W         W
        WWWWWWWWWWW
        """)
        self.assert_step_sequence(
            bd, """
            WWWWWWWWWWW
            W         W
            W R RW WWRW
            W         W
            WWWWWWWWWWW
            """, """
            WWWWWWWWWWW
            W         W
            W    W WW W
            W R R    RW
            WWWWWWWWWWW
            """, """
            WWWWWWWWWWW
            W         W
            W    W WW W
            W R R    RW
            WWWWWWWWWWW
            """
        )

    def test_301_step(self):
        bd = BoulderDash.from_string_map("""
        RR
        ..
        """)
        for i in range(2):
            bd.step()
            self.assert_map(bd, """
            ..
            RR
            """)

    def test_310_step_successive_falling(self):
        bd = BoulderDash.from_string_map("""
        .RDRDRDRDR.
        .WRDRDRDRD.
        . W  W WW .
        .   W     .
        ...........
        """)
        self.assert_step_sequence(
            bd, """
            . D  R RD .
            RWRRDDDDRR.
            . WDRWRWWD.
            .   W     .
            ...........
            """, """
            .      R  .
            .WRDDDRDRD.
            R WRRWDWWR.
            .  DW R  D.
            ...........
            """, """
            .         .
            .WR DDRDR .
            . WDRWRWWD.
            R  RW D  R.
            ...D..R..D.
            """, """
            .         .
            .W  DD D  .
            . WRRWRWWR.
            .  DW R  D.
            R.RD.DR.RD.
            """, """
            .         .
            .W  D  D  .
            . WRRWDWW .
            .  DW R  R.
            R.RD.DRRRDD
            """, """
            .         .
            .W  D  D  .
            . WRRWDWW .
            .  DW R  R.
            R.RD.DRRRDD
            """
        )

    def test_500_all(self):
        bd = BoulderDash.from_string_map("""
        WWWWWWWWWWW
        WD        W
        WWR       W
        WPSS      W
        WWWWWWWWWWW
        """)
        self.assertEqual(bd.RESULTS.RemovedSand, bd.apply_action(bd.ACTIONS.Right))
        self.assertEqual(bd.step(), bd.RESULTS.Nothing)
        self.assert_map(bd, """
        WWWWWWWWWWW
        WD        W
        WWR       W
        W PS      W
        WWWWWWWWWWW
        """)
        self.assertEqual(bd.RESULTS.Blocked, bd.apply_action(bd.ACTIONS.Up))
        self.assertEqual(bd.step(), bd.RESULTS.Nothing)
        self.assert_map(bd, """
        WWWWWWWWWWW
        WD        W
        WWR       W
        W PS      W
        WWWWWWWWWWW
        """)
        self.assertEqual(bd.RESULTS.RemovedSand, bd.apply_action(bd.ACTIONS.Right))
        self.assertEqual(bd.step(), bd.RESULTS.Nothing)
        self.assert_map(bd, """
        WWWWWWWWWWW
        W         W
        WWD       W
        W RP      W
        WWWWWWWWWWW
        """)
        self.assertEqual(bd.RESULTS.Moved, bd.apply_action(bd.ACTIONS.Right))
        self.assertEqual(bd.step(), bd.RESULTS.Nothing)
        self.assert_map(bd, """
        WWWWWWWWWWW
        W         W
        WW        W
        W RDP     W
        WWWWWWWWWWW
        """)
        self.assertEqual(bd.RESULTS.Nothing, bd.apply_action(bd.ACTIONS.Nop))
        self.assertEqual(bd.step(), bd.RESULTS.Nothing)
        self.assert_map(bd, """
        WWWWWWWWWWW
        W         W
        WW        W
        W RDP     W
        WWWWWWWWWWW
        """)
        self.assertEqual(bd.RESULTS.CollectedAllDiamonds, bd.apply_action(bd.ACTIONS.Left))
        self.assertEqual(bd.step(), bd.RESULTS.Nothing)
        self.assert_map(bd, """
        WWWWWWWWWWW
        W         W
        WW        W
        W RP      W
        WWWWWWWWWWW
        """)
