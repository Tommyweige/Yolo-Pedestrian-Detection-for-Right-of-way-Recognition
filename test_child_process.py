import sys
import unittest

from child_process import TAIL_LINES, Child


class ChildTest(unittest.TestCase):
    def test_lines_arrive_in_order_and_the_exit_code_and_tail_are_kept(self):
        child = Child([sys.executable, "-c", "import sys\nfor i in range(20): print(i)\nsys.exit(4)"])
        self.assertEqual([line.strip() for line in child.lines()], [str(i) for i in range(20)])
        self.assertEqual(child.code, 4)
        self.assertFalse(child.stopped)
        self.assertEqual(child.tail, [str(i) for i in range(20 - TAIL_LINES, 20)])

    def test_a_chatty_child_does_not_block_on_a_full_pipe(self):
        child = Child([sys.executable, "-c", "for i in range(20000): print('x' * 80)"])
        self.assertEqual(sum(1 for _ in child.lines()), 20000)
        self.assertEqual(child.code, 0)

    def test_cancelling_terminates_the_child(self):
        child = Child([sys.executable, "-c", "print('started', flush=True)\nimport time; time.sleep(60)"],
                      cancelled=lambda: True)
        list(child.lines())
        self.assertTrue(child.stopped)
        self.assertNotEqual(child.code, 0)

    def test_an_abandoned_loop_still_stops_the_child_and_releases_it(self):
        seen = []
        child = Child([sys.executable, "-c", "print('one', flush=True)\nimport time; time.sleep(60)"],
                      on_process=seen.append)
        lines = child.lines()
        self.assertEqual(next(lines).strip(), "one")
        lines.close()
        self.assertIsNotNone(seen[0].returncode)
        self.assertIsNone(seen[-1])
        self.assertIsNone(child.code)  # it never finished on its own


if __name__ == "__main__":
    unittest.main()
