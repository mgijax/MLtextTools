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

    def test_distinctMappingNames(self):
        dupMapping = TextMapping('THE', r'this is a duplicate name', 'abc')
        with self.assertRaises(ValueError):
            t = TextTransformer(self.THEmappings + [dupMapping])

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

class HelperFunction_tests(unittest.TestCase):

    def test_escAndWordBoundaries(self):
        s = r'some[a][b]*\word\s'
        regex = escAndWordBoundaries(s)
        #print()
        #print("'%s'" % s)
        #print("'%s'" % regex)
        r = re.compile(regex)
        self.assertIsNotNone(r.search(s))               # should match itself
        self.assertIsNotNone(r.search('word ' + s + '.fun'))
        self.assertIsNone(r.search('some text'))

    def test_squeezeAndEscape(self):
        s  = 'some word\\s  more\nwords'
        s2 = 'some     word\\s  more  \twords'
        regex = squeezeAndEscape(s)
        #print()
        #print("'%s'" % s)
        #print("'%s'" % regex)
        r = re.compile(regex)
        self.assertIsNotNone(r.search(s))
        self.assertIsNotNone(r.search(s2))
        self.assertIsNotNone(r.search('word ' + s2 + '.fun'))

# end class TextMappingFromStrings_tests
######################################

class TextMappingFromStrings_tests(unittest.TestCase):

    def test_FromStringsBasic(self):
        strings = [r'abc', r'word (in) parens',]
        tm = TextMappingFromStrings('myname', strings, 'foo', context=3)
        t = TextTransformer([tm])
        #print()
        #print("'%s'" % tm.regex)
        #print("'%s'" % squeezeAndEscape(strings[0]))

        text = 'start abc, abcdef. Word (in) parens. end'
        expect = 'start foo, abcdef. foo. end'
        transformed = t.transformText(text)
        self.assertEqual(transformed, expect)
        #print('\n' + t.getsReport())

# end class TextMappingFromStrings_tests
######################################

class TextMappingFromFile_tests(unittest.TestCase):

    def setUp(self):
        # write a test mapping file
        self.fileName = 'mappingFromFileTestData.txt'
        lines = ['1246', 'a[1]b', '14F1.1', ' two  words ', '# foo']
        with open(self.fileName, 'w') as fp:
            for t in lines:
                fp.write(t + '\n')
        self.tm = TextMappingFromFile('basic', self.fileName, 'NEW')

    def tearDown(self):
        os.remove(self.fileName)
        pass

    def test_FromFileBasic(self):
        t = TextTransformer([self.tm])
        #print('\n')
        #print("'%s'" % self.tm.regex)
        text = 'start 1246, end'
        expect = 'start NEW, end'
        self.assertEqual(t.transformText(text), expect)

        text = 'start A[1]b, end'
        expect = 'start NEW, end'
        self.assertEqual(t.transformText(text), expect)

        text = 'start 14f1.1, end'      # '.' should match '.'
        expect = 'start NEW, end'
        self.assertEqual(t.transformText(text), expect)

        text = 'start 14f1:1, end'      # '.' should not match ':'
        expect = 'start 14f1:1, end'    # no change
        self.assertEqual(t.transformText(text), expect)

        text = 'start:two words, end'   # handle arbitrary spaces
        expect = 'start:NEW, end'
        self.assertEqual(t.transformText(text), expect)

        text = 'start:two      words end'       # handle arbitrary spaces
        expect = 'start:NEW end'
        self.assertEqual(t.transformText(text), expect)

        text = 'start foo end'
        expect = 'start foo end'        # no change since foo commented out
        self.assertEqual(t.transformText(text), expect)

        #print('\n' + t.getsReport())

# end class TextMappingFromFile_tests
######################################

if __name__ == '__main__':
    unittest.main()
