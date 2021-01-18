#!/usr/bin/env python3

import sys
import unittest
import os
import os.path
from baseSampleDataLib import *

"""
These are tests for baseSampleDataLib.py

Usage:   python test_baseSampleDataLib.py [-v]

The pattern:
    Each class in baseSampleDataLib.py has a corresponding *_tests class here to
    exercise methods.
"""
######################################

class BaseSample_tests(unittest.TestCase):
    def setUp(self):
        self.sample1Text = '''pmID1|text1'''
        self.sample1 = BaseSample().parseSampleRecordText(self.sample1Text)
        self.sample2 = BaseSample().parseSampleRecordText('''pmID2|text2''')

    def test_setgetFields(self):
        s = BaseSample()
        d = {'ID': 'pmID4', 'text': 'text4', 'foo': 'invalid field'}
        s.setFields(d)
        self.assertEqual('pmID4', s.getField('ID'))
        self.assertEqual('text4', s.getField('text'))
        self.assertRaises(KeyError, s.getField, 'journal')
        self.assertRaises(KeyError, s.getField, 'foo')

    def test_setgetID(self):
        self.assertEqual(self.sample1.getID(), 'pmID1')
        self.sample1.setID('pmID25')
        self.assertEqual(self.sample1.getID(), 'pmID25')

    def test_getSampleName(self):
        self.assertEqual(self.sample1.getSampleName(), 'pmID1')

    def test_getSampleID(self):
        self.assertEqual(self.sample1.getSampleID(), 'pmID1')

    def test_getDocument(self):
        self.assertEqual('text1', self.sample1.getDocument())

    def test_getSampleAsText(self):
        self.assertEqual(self.sample1.getSampleAsText(), self.sample1Text)

    def test_getFieldNames(self):
        self.assertEqual(['ID', 'text'], self.sample1.getFieldNames())

    def test_getClassNames(self):
        self.assertEqual(['no', 'yes'], self.sample1.getClassNames())

    def test_getY_positive(self):
        self.assertEqual(1, self.sample1.getY_positive())

    def test_getY_negative(self):
        self.assertEqual(0, self.sample1.getY_negative())

    def test_getHeaderLine(self):
        self.assertIn('ID', self.sample1.getHeaderLine())
        self.assertIn('text', self.sample1.getHeaderLine())

    def test_getFieldSep(self):
        self.assertEqual('|', self.sample1.getFieldSep())

    def test_removeURLsLower(self):
        sample3 = BaseSample().parseSampleRecordText(\
                            '''pmID3|Some Text http://url.org end text''')
        expectedText = "some text   end text"
        sample3.removeURLsLower()
        self.assertEqual(expectedText, sample3.getDocument())

    def test_tokenPerLine(self):
        sample3 = BaseSample().parseSampleRecordText(\
                            'pmID3|Some Text,\nhttp://url.org 123 _abc_123\n')
        expectedText = "Some\nText\nhttp\nurl\norg\n123\n_abc_123\n"
        sample3.tokenPerLine()
        self.assertEqual(expectedText, sample3.getDocument())

    def test_truncateText(self):
        sample3 = BaseSample().parseSampleRecordText(\
                            'pmID3|1234567890123456789012345\n')
        expectedText = "12345678901234567890\n"
        sample3.truncateText()
        self.assertEqual(expectedText, sample3.getDocument())
# end class BaseSample_tests
######################################

class ClassifiedSample_tests(unittest.TestCase):
    def setUp(self):
        self.sample1Text = '''no|pmID1|text1'''
        self.sample1 = ClassifiedSample().parseSampleRecordText( \
                                                           self.sample1Text)
        self.sample2 = ClassifiedSample().parseSampleRecordText( \
                                                        '''yes|pmID2|text2''')

    def test_setgetFields(self):
        s = ClassifiedSample()
        d = {'knownClassName': 'yes', 'ID': 'pmID4', 'text': 'text4',
                'foo': 'invalid field'}
        s.setFields(d)
        self.assertEqual('pmID4', s.getField('ID'))
        self.assertEqual('yes', s.getField('knownClassName'))
        self.assertEqual('text4', s.getField('text'))
        self.assertRaises(KeyError, s.getField, 'foo')

    def test_setgetID(self):
        self.assertEqual(self.sample1.getID(), 'pmID1')
        self.sample1.setID('pmID25')
        self.assertEqual(self.sample1.getID(), 'pmID25')

    def test_getSampleName(self):
        self.assertEqual(self.sample1.getSampleName(), 'pmID1')

    def test_getSampleID(self):
        self.assertEqual(self.sample1.getSampleID(), 'pmID1')

    def test_getDocument(self):
        self.assertEqual(self.sample1.getDocument(), 'text1')

    def test_getSampleAsText(self):
        self.assertEqual(self.sample1.getSampleAsText(), self.sample1Text)

    def test_getFieldNames(self):
        self.assertEqual(['knownClassName', 'ID', 'text'], \
                                                self.sample1.getFieldNames())

    def test_getClassNames(self):
        self.assertEqual(['no', 'yes'], self.sample1.getClassNames())

    def test_getY_positive(self):
        self.assertEqual(1, self.sample1.getY_positive())

    def test_getY_negative(self):
        self.assertEqual(0, self.sample1.getY_negative())

    def test_getHeaderLine(self):
        self.assertIn('ID', self.sample1.getHeaderLine())

    def test_getFieldSep(self):
        self.assertEqual('|', self.sample1.getFieldSep())

    def test_setgetKnownClassName(self):
        self.assertEqual(self.sample1.getKnownClassName(), 'no')
        self.assertEqual(self.sample2.getKnownClassName(), 'yes')

        self.assertRaises(ValueError, self.sample1.setKnownClassName, 'bad')
        self.sample1.setKnownClassName(';no')
        self.assertEqual(self.sample1.getKnownClassName(), 'no')
        self.sample1.setKnownClassName(';yes')
        self.assertEqual(self.sample1.getKnownClassName(), 'yes')

    def test_getKnownYvalue(self):
        self.assertEqual(self.sample1.getKnownYvalue(), 0)
        self.assertEqual(self.sample2.getKnownYvalue(), 1)

    def test_isPositiveNegative(self):
        self.assertFalse(self.sample1.isPositive())
        self.assertTrue( self.sample1.isNegative())
        self.assertTrue( self.sample2.isPositive())
        self.assertFalse(self.sample2.isNegative())

    def test_getExtraInfo(self):
        self.assertEqual([], self.sample1.getExtraInfoFieldNames())
        self.assertEqual([], self.sample1.getExtraInfo())

# end class ClassifiedSample_tests
######################################

class SampleSetMetaData_tests(unittest.TestCase):
    def setUp(self):
        line1 = "#meta foo=blah   nose=toes     rose=rose"
        self.meta1 = SampleSetMetaData(line1)

    def test_hasMetaData(self):
        self.assertTrue(self.meta1.hasMetaData())
        self.assertTrue(self.meta1)    # test m as a __bool__

        m = SampleSetMetaData("#text with no\nmeta line")
        self.assertFalse(m.hasMetaData())
        self.assertFalse(m)            # test m as a __bool__

    def test_setgetMetaDict(self):
        d = {'foo': '1', 'blah': 'abc'}
        self.meta1.setMetaDict(d)
        self.assertEqual(d, self.meta1.getMetaDict())

    def test_setgetMetaItem(self):
        self.assertEqual('toes', self.meta1.getMetaItem('nose'))
        self.meta1.setMetaItem('nose', 'tomatoes')
        self.assertEqual('tomatoes', self.meta1.getMetaItem('nose'))

    def test_buildMetaLine(self):
        m = SampleSetMetaData('')
        m.setMetaDict({'foo': 'blah'})
        expectedText = '#meta foo=blah'
        self.assertEqual(expectedText, m.buildMetaLine())

# end class SampleSetMetaData_tests
######################################

class SampleSet_tests(unittest.TestCase):
    def setUp(self):
        self.ss = SampleSet(sampleObjType=BaseSample)
        self.sample1Text =  '''pmID1|text1'''
        self.sample1 = BaseSample().parseSampleRecordText(self.sample1Text)
        self.sample2 = BaseSample().parseSampleRecordText('pmID2|text2')

        self.ss.addSamples([self.sample1, self.sample2]) # exercise addSamples()

    def test_addSampleTypeError(self):
        self.assertRaises(TypeError, self.ss.addSample, ClassifiedSample())

    def test_getSamples(self):
        self.assertEqual(self.sample1, self.ss.getSamples()[0])
        self.assertEqual(self.sample2, self.ss.getSamples()[1])

    def test_getSampleIDs(self):
        self.assertEqual(['pmID1', 'pmID2'], self.ss.getSampleIDs())

    def test_getDocuments(self):
        self.assertEqual(['text1', 'text2'], self.ss.getDocuments())

    def test_getNumSamples(self):
        self.assertEqual(2, self.ss.getNumSamples())

    def test_getRecordEnd(self):
        self.assertEqual(';;', self.ss.getRecordEnd())

    def test_getSampleObjType(self):
        self.assertEqual(BaseSample,self.ss.getSampleObjType())

    def test_getSampleClassNames(self):
        self.assertEqual(['no','yes'], self.ss.getSampleClassNames())

    def test_getY_positive(self):
        self.assertEqual(1, self.ss.getY_positive())

    def test_getY_negative(self):
        self.assertEqual(0, self.ss.getY_negative())

    def test_getFieldNames(self):
        self.assertEqual(['ID', 'text'], self.ss.getFieldNames())

    def test_getHeaderLine(self):
        self.assertIn('ID', self.ss.getHeaderLine())
        self.assertIn('text',    self.ss.getHeaderLine())

    def test_write_read(self):
        fileName = 'temporarySampleOutputFile.txt'
        fp = open(fileName, 'w')
        self.ss.write(fp)
        fp.close()

        # read in the file, verify things seem identical
        fp = open(fileName, 'r')
        ss2 = SampleSet().read(fp)
        fp.close()
        self.assertEqual(self.ss.getSampleObjType(), ss2.getSampleObjType())
        self.assertEqual(self.ss.getHeaderLine(), ss2.getHeaderLine())
        self.assertEqual(self.ss.getDocuments()[0], ss2.getDocuments()[0])
        self.assertEqual(self.ss.getDocuments()[1], ss2.getDocuments()[1])

        os.remove(fileName)

    def test_rejection(self):
        # test SampleSet before rejecting any samples
        self.assertEqual(2, self.ss.getNumSamples(omitRejects=True))
        self.assertEqual([self.sample1, self.sample2],
                                self.ss.getSamples(omitRejects=True))
        self.assertEqual(['pmID1', 'pmID2'],
                                self.ss.getSampleIDs(omitRejects=True))
        self.assertEqual(['text1', 'text2'],
                                self.ss.getDocuments(omitRejects=True))
        # Reject one sample
        self.sample1.setReject(True, reason='my rejection reason')
        self.assertTrue(self.sample1.isReject())
        self.assertEqual('my rejection reason', self.sample1.getRejectReason())

        # test SampleSet again
        self.assertEqual(1, self.ss.getNumSamples(omitRejects=True))
        self.assertEqual([self.sample2], self.ss.getSamples(omitRejects=True))
        self.assertEqual(['pmID2'], self.ss.getSampleIDs(omitRejects=True))
        self.assertEqual(['text2'], self.ss.getDocuments(omitRejects=True))

    def test_preprocess(self):
        sample3 = BaseSample().parseSampleRecordText(\
                            '''pmID3|Some Text http://url.org end text''')
        expectedText = "some\ntext\nend\ntext\n"
        self.ss.addSample(sample3)
        self.ss.preprocess(['removeURLsLower', 'tokenPerLine'])
        self.assertEqual(expectedText, self.ss.getDocuments()[2])

# end class SampleSet_tests
######################################

class ClassifiedSampleSet_tests(unittest.TestCase):
    def setUp(self):
        # build sampleSet w/ 3 samples
        self.ss = ClassifiedSampleSet(sampleObjType=ClassifiedSample)
        self.sample1Text = \
        '''no|pmID1|text1'''
        self.sample1 = ClassifiedSample().parseSampleRecordText(\
                                                    self.sample1Text)
        self.sample2 = ClassifiedSample().parseSampleRecordText(\
                                                    '''yes|pmID2|text2''')
        self.sample3 = ClassifiedSample().parseSampleRecordText(\
                                                    '''no|pmID3|text3''')

        self.ss.addSamples([self.sample1, self.sample2]) # exercise addSamples()
        self.ss.addSample(self.sample3)                  # exercise addSample()

    #### Methods inherited from SampleSet
    def test_addSampleTypeError(self):
        self.assertRaises(TypeError, self.ss.addSample, BaseSample())

    def test_getSamples(self):
        self.assertEqual(self.sample1, self.ss.getSamples()[0])
        self.assertEqual(self.sample2, self.ss.getSamples()[1])
        self.assertEqual(self.sample3, self.ss.getSamples()[2])

    def test_getSampleIDs(self):
        self.assertEqual(['pmID1', 'pmID2', 'pmID3'], self.ss.getSampleIDs())

    def test_getDocuments(self):
        self.assertEqual(['text1', 'text2', 'text3'], self.ss.getDocuments())

    def test_getNumSamples(self):
        self.assertEqual(3, self.ss.getNumSamples())

    def test_getRecordEnd(self):
        self.assertEqual(';;', self.ss.getRecordEnd())

    def test_getSampleObjType(self):
        self.assertEqual(ClassifiedSample, self.ss.getSampleObjType())

    def test_getSampleClassNames(self):
        self.assertEqual(['no','yes'], self.ss.getSampleClassNames())

    def test_getY_positive(self):
        self.assertEqual(1, self.ss.getY_positive())

    def test_getY_negative(self):
        self.assertEqual(0, self.ss.getY_negative())

    def test_getFieldNames(self):
        self.assertEqual(['knownClassName','ID','text'],self.ss.getFieldNames())

    def test_getHeaderLine(self):
        # not bothering to check for all fields, just a few
        self.assertIn('knownClassName', self.ss.getHeaderLine())
        self.assertIn('text', self.ss.getHeaderLine())

    def test_write_read(self):
        fileName = 'temporarySampleOutputFile.txt'
        fp = open(fileName, 'w')
        self.ss.write(fp)
        fp.close()

        # read in the file, verify things seem identical
        fp = open(fileName, 'r')
        ss2 = ClassifiedSampleSet().read(fp)
        fp.close()
        self.assertEqual(self.ss.getSampleObjType(), ss2.getSampleObjType())
        self.assertEqual(self.ss.getHeaderLine(),    ss2.getHeaderLine())
        self.assertEqual(self.ss.getDocuments()[0],  ss2.getDocuments()[0])
        self.assertEqual(self.ss.getDocuments()[1],  ss2.getDocuments()[1])
        self.assertEqual(self.ss.getDocuments()[2],  ss2.getDocuments()[2])

        os.remove(fileName)

    def test_rejection(self):

        # test SampleSet before rejecting any samples
        self.assertEqual(3, self.ss.getNumSamples(omitRejects=True))
        self.assertEqual([self.sample1, self.sample2, self.sample3],
                                self.ss.getSamples(omitRejects=True))
        self.assertEqual(['pmID1', 'pmID2', 'pmID3'],
                                self.ss.getSampleIDs(omitRejects=True))
        self.assertEqual(['no', 'yes', 'no'],
                                self.ss.getKnownClassNames(omitRejects=True))
        self.assertEqual(['text1', 'text2', 'text3'],
                                self.ss.getDocuments(omitRejects=True))
        # Reject one sample
        self.sample1.setReject(True, reason='my rejection reason')
        self.assertTrue(self.sample1.isReject())
        self.assertEqual('my rejection reason', self.sample1.getRejectReason())

        # test SampleSet again
        self.assertEqual(2, self.ss.getNumSamples(omitRejects=True))
        self.assertEqual([self.sample2, self.sample3],
                                self.ss.getSamples(omitRejects=True))
        self.assertEqual(['pmID2', 'pmID3'],
                                self.ss.getSampleIDs(omitRejects=True))
        self.assertEqual(['text2', 'text3'],
                                self.ss.getDocuments(omitRejects=True))

    ### Methods from ClassifiedSampleSet
    def test_getKnownClassNames(self):
        self.assertEqual(['no', 'yes', 'no'], self.ss.getKnownClassNames())

    def test_getKnownYvalues(self):
        self.assertEqual([0, 1, 0], self.ss.getKnownYvalues())

    def test_getNumPositives(self):
        self.assertEqual(1, self.ss.getNumPositives())

    def test_getNumNegatives(self):
        self.assertEqual(2, self.ss.getNumNegatives())

    def test_getExtraInfoFieldNames(self):
        self.assertEqual([], self.ss.getExtraInfoFieldNames())

# end class ClassifiedSampleSet_tests
######################################

if __name__ == '__main__':
    unittest.main()
