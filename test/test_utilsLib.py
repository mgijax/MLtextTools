#!/usr/bin/env python3

import unittest
from utilsLib import *

"""
These are tests for utilsLib.py

Usage:   python test_utilsLib.py [-v]
"""
######################################

class TextTransformer_tests(unittest.TestCase):
    def setUp(self):
        self.THEmappings = [
            TextMapping('THE', r'\b(?:the)\b', 'the_'),
            TextMapping('THESE', r'\b(?:these)\b', 'these_'),
        ]

    def test__buildBigRe(self):
        t = TextTransformer(self.THEmappings)
        #print(t.getBigRegex())
        self.assertEqual(t.getBigRegex(), t.getBigRe().pattern)

    def test_transformText(self):
        t = TextTransformer(self.THEmappings)
        text = "there are These things & these & these, and then the end"
        done = "there are these_ things & these_ & these_, and then the_ end"
        transformed = t.transformText(text)
        #print("\n'%s'" % transformed)
        self.assertEqual(transformed, done)

        text = "here\nis some random junk !@#$%^&**"    # no matches
        self.assertEqual(text, t.transformText(text))
        self.assertEqual('', t.transformText(''))       # empty string

    def test_getMatches(self):
        t = TextTransformer(self.THEmappings)
        text = "there are These things & these & these, and then the end"
        transformed = t.transformText(text)
        matches = t.getMatches()
        #print(matches)
        self.assertEqual(matches[1][1]['these'], 2)
        self.assertEqual(matches[1][1]['These'], 1)

        transformed = t.transformText(text)             # should update counts
        matches = t.getMatches()
        self.assertEqual(matches[1][1]['these'], 4)
        self.assertEqual(matches[1][1]['These'], 2)

    def test_resetMatches(self):
        t = TextTransformer(self.THEmappings)
        text = "there are These things & these & these, and then the end"
        transformed = t.transformText(text)
        matches = t.getMatches()
        self.assertEqual(matches[1][1]['these'], 2)
        self.assertEqual(matches[1][1]['These'], 1)

        t.resetMatches()                                # should reset counts
        transformed = t.transformText(text)
        matches = t.getMatches()
        self.assertEqual(matches[1][1]['these'], 2)
        self.assertEqual(matches[1][1]['These'], 1)

# end class TextTransformer_tests
######################################

if __name__ == '__main__':
    unittest.main()
