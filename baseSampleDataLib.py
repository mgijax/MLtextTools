#!/usr/bin/env python3
#
# Library to support handling of text machine learning samples and sample files
#    (training samples or samples to predict)
#
# Automated tests for this module are available:
#  cd test
#  python test_baseSampleDataLib.py -v
#
import sys
import os.path
import string
import re
from copy import copy
import inspect
import utilsLib

#-----------------------------------
#
# Naming conventions:
#  * use camelCase for most things
#  * but stick to sklearn convention for y_*  which are the indexes of
#      sample classification names, e.g., 'discard', 'keep'
#  * use "class" or "classname" to mean the sample classification names
#      (e.g., so a ClassifiedSample is a sample where we know if it is
#       'discard' or 'keep')
#  * this is confusing with python "classes" and with the fact that the name
#      of a python class is passed as a config parameter to specify which
#      ClassifiedSample subclass to instantiate.
#  * so try to use "python class" or "object type" for these
#-----------------------------------
Class_Hierarchy_Overview = \
"""
In baseSampleDataLib.py (MLTextTools)
    BaseSample
        - a text sample (classified or not) for a binary ML problem.
        - knows the two class names for the problem (e.g., "no", "yes"),
            their Y values (0/1), & which is considered the "positive" class.
        - has an arbitrary list of fields
            "ID" must be one of these
            "text" is a field by default. Can be overridden in subclasses.
        - getDocument() returns the text to be consider by a classifier
            (the "text" field by default, but subclasses may combine multiple
             fields into the "document", e.g., title, abstract, extractedText)
        - a sample may be "rejected" and have a rejection reason. Idea:
            During preprocessing, we may decide that a sample is not suitable
            for training or classification
            (relevanceClassifier doesn't use this feature)
        - implements converting a Sample to and from text for reading/writing
        - Samples can have preprocessing methods that modify the state of 
            the sample (e.g., stemming the text)
        - BaseSample implements some generic preprocessing primitives
    ClassifiedSample
        - a text sample that is classified (has a knownClassName, Y value)
        - has an optional set of ExtraInfo fields: additional info about the
            sample that is not meant to be used by a classifier but is useful
            for analyzing classifier results
    SampleSet
        - a collection of Samples of the same type (BaseSample or descendent)
        - reads/writes Sample files incl. optional meta data
        - get parallel lists:    getSamples(), getSampleIDs(), getDocuments() 
    ClassifiedSampleSet
        - a SampleSet of ClassifiedSamples
        - get parallel lists:    getKnownClassNames(), getKnownYvalues()
        - getExtraInfoFieldNames()
    SampleSetMetaData
        - info about the Samples in a Sample file
        - most important: name of the SampleObjType (python class name)
            & the name of the python module that defines that type
                (new in Jan 2021, previous versions assumed "sampleDataLib")
        - enables SampleSet to read/write sets of different types of Samples
        - meta data in a sample file is still optional for backward
            compatability, but it would be simpler to make it required at this
            point
"""

FIELDSEP  = '|'      # dflt field separator when reading/writing sample fields
RECORDEND = ';;'     # dflt record ending str when reading/writing sample files

#-----------------------------------

class BaseSample (object):
    """
    Base class for text samples that my be classified or unclassified.

    HAS: ID, text
        Subclasses can have other fields (and omit the text field) defined
        in fieldNames below

        An ID field is required.

        Has the class names (e.g., 'no', 'yes'), knows which class is
            considered "positive", etc.
    DOES:
        Can read/write the sample as text
        Provides various methods to preprocess a sample record
        (preprocess the text prior to vectorization)
        Samples can be "rejected" and have a rejection reason.
    """
                # I think these need to be in alpha order if you
                #  load samples using sklearn's load_files() function.
                #  (and they need to match the directory names where the
                #  sample files are stored)
    sampleClassNames = ['no','yes']
    y_positive = 1	# sampleClassNames[y_positive] is the "positive" class
    y_negative = 0	# ... negative

    # fields of a sample as an input/output record (as text), in order
    fieldNames = [ \
            'ID'  ,
            'text',
            ]
    fieldSep  = FIELDSEP
    recordEnd = RECORDEND

    def __init__(self,):
        self.isRejected = False
        self.rejectReason = None
    #----------------------

    def parseSampleRecordText(self, text):
        """
        Parse the text representing a sample record and populate self
        with that record
        """
        values = {}
        fields = text.split(self.fieldSep)

        for i, fn in enumerate(self.fieldNames):
            values[fn] = fields[i]

        return self.setFields(values)
    #----------------------

    def getSampleAsText(self):
        """ Return this sample as a text string
        """
        return self.fieldSep.join([ self.values[fn] for fn in self.fieldNames])
    #----------------------

    def setFields(self, values,		# dict
        ):
        """
        Set the fields of this sample from a dictionary of field values.
        If the dict does not have a value for a field, it defaults to ''
        """
        self.values = { fn: str(values.get(fn,'')) for fn in self.fieldNames }
        return self
    #----------------------

    def setField(self, fieldName, value):
        self.values[fieldName] = str(value)

    def getField(self, fieldName):
        return self.values[fieldName]
    #----------------------

    def constructDoc(self):
        """ 
        Return the text of the "document" of this sample, i.e., the
            string that a classifier should consider.
        Override this method if your samples don't have a simple "text" field
        """
        return self.values['text']

    def getDocument(self):	return self.constructDoc()
    #----------------------

    def setID(self, t):		self.values['ID'] = t
    def getID(self,  ):		return self.values['ID']
    def getSampleName(self):	return self.getID()
    def getSampleID(self):	return self.getID()
    def getName(self):		return self.getID()
    #----------------------

    @classmethod
    def getHeaderLine(cls):	return cls.fieldSep.join( cls.fieldNames)

    @classmethod
    def getFieldSep(cls):	return cls.fieldSep

    @classmethod
    def getRecordEnd(cls):	return cls.recordEnd

    @classmethod
    def getFieldNames(cls):	return cls.fieldNames

    @classmethod
    def getClassNames(cls):	return cls.sampleClassNames

    @classmethod
    def getY_positive(cls):	return cls.y_positive

    @classmethod
    def getY_negative(cls):	return cls.y_negative

    #----------------------
    def setReject(self, value, reason=None):
        self.isRejected = value         # value should be True or False
        self.rejectReason = reason

    def isReject(self):	
        return self.isRejected
    def getRejectReason(self):	return self.rejectReason

    #----------------------
    # "preprocessor" functions.
    #  Each preprocessor should modify this sample and return itself
    #----------------------

    def removeURLsLower(self):		# preprocessor
        '''
        Remove URLs, lower case everything,
        '''
        self.setField('text', utilsLib.removeURLsLower(self.getField('text')))
        return self
    # ---------------------------

    def tokenPerLine(self):		# preprocessor
        """
        Convert text to have one alphanumeric token per line,
            removing punctuation.
        Makes it easier to examine the tokens/features
        """
        self.setField('text', utilsLib.tokenPerLine(self.getField('text')))
        return self
    # ---------------------------

    def truncateText(self):		# preprocessor
        """ for debugging, so you can see a sample record easily"""
        
        self.setField('text', self.getField('text')[:20].replace('\n',' ')+'\n')
        return self
    # ---------------------------
# end class BaseSample ------------------------

class ClassifiedSample (BaseSample):
    """
    A BaseSample that is classified (has a knownClassName, Y value)
        - has an optional set of ExtraInfo fields: additional info about the
            sample that is not meant to be used by a classifier but is useful
            for analyzing classifier results
    """
    fieldNames = [ \
            'knownClassName',
            'ID'            ,
            'text'          ,
            ]
    extraInfoFieldNames = [  ] # should be [] if no extraInfoFields
    #----------------------

    def setFields(self, values,		# dict
        ):
        BaseSample.setFields(self,values)
        self.setKnownClassName( values['knownClassName'] )
        return self
    #----------------------

    def setKnownClassName(self, t):
        self.values['knownClassName'] = self.validateClassName(t)
        
    className_re = re.compile(r'\b(\w+)\b')	# all alpha numeric
    @classmethod
    def validateClassName(cls, className):
        """
        1) validate className is a sampleClassName
        2) transform it as needed: remove any leading/trailing spaces and punct
        Return the cleaned up name, or raise ValueError.
        The orig need for cleaning up arose when using ';;' as the record sep
            and having some extracted text ending in ';'.
            So splitting records on ';;' left the record's class as ';discard'
            which caused problems down the line.
        """
        m = cls.className_re.search(className)

        if m and m.group() in cls.sampleClassNames:
            return m.group()
        else:
            raise ValueError("Invalid sample classification '%s'\n" % \
                                                                str(className))
    #----------------------

    def getKnownClassName(self): return self.values['knownClassName']
    def getKnownYvalue(self):
        return self.sampleClassNames.index(self.getKnownClassName())
    def isPositive(self):
        return self.getKnownYvalue() == self.y_positive 
    def isNegative(self):
        return not self.isPositive()
    #----------------------
    @classmethod
    def getExtraInfoFieldNames(cls): return cls.extraInfoFieldNames
    def getExtraInfo(self):
        self.extraInfo = { fn : str(self.values.get(fn,'none')) \
                                    for fn in self.getExtraInfoFieldNames() }
        self.setComputedExtraInfoFields()
        return [ self.extraInfo[x] for x in self.getExtraInfoFieldNames() ]
        
    def setComputedExtraInfoFields(self):
        """ override this if you want to compute any extra info fields, e.g.,
            self.extraInfo['textlen'] = str(len(self.getField('text')))
        """
        pass
    #----------------------
# end class ClassifiedSample ------------------------

#-----------------------------------
# SampleSets
#-----------------------------------

class SampleSet (object):
    """
    IS:     a set of Samples
    HAS:    list of Samples, list of Sample IDs, list of Sample documents, ...
    DOES:   Loads/parses sample records from a sample record file
            Writes sample records to a file
    """
    def __init__(self, sampleObjType=None,
        ):
        self.sampleObjType = sampleObjType
        self.meta = None        # optional metadata from sample set file

        if self.sampleObjType:
            self.recordEnd = self.sampleObjType.getRecordEnd()
        else:
            self.recordEnd = RECORDEND
        self.samples = []
    #-------------------------

    def read(self, inFile,	# file pathname or open file obj for reading
        ):
        """
        Assumes sample record file is not empty and has header text
        """
        if type(inFile) == type(''): fp = open(inFile, 'r')
        else: fp = inFile

        self.textToSamples(fp.read())

        if type(inFile) == type(''): fp.close() # close if we opened it
        
        return self
    #-------------------------

    def textToSamples(self, text,
        ):
        self.meta = SampleSetMetaData()
        text = self.meta.consumeMetaText(text)

        if self.meta.hasMetaData():
            sampleObjTypeName = self.meta.getMetaItem('sampleObjType')
            if sampleObjTypeName:       # sample obj type is in meta
                # Note sampleObjType in the file overrides what is passed
                #  during instantiation.
                # get module
                moduleName = self.meta.getMetaItem('moduleName')
                if not moduleName:      # module not in meta
                    lastTry = 'sampleDataLib'
                    if hasattr(sys.modules[__name__], sampleObjTypeName):
                        moduleName = __name__           # in current module
                    elif hasattr(sys.modules[lastTry], sampleObjTypeName):
                        moduleName = lastTry
                    else:
                        t = "Cannot find module that defines '%s'\n" \
                                                        % sampleObjTypeName
                        raise(NameError(t))

                self.sampleObjType = getattr(sys.modules[moduleName],
                                                            sampleObjTypeName)
                self.recordEnd = self.sampleObjType.getRecordEnd()
            # else: assume sample obj type was set upon instantiation

        rcds = text.split(self.recordEnd)
        del rcds[0]             # header text
        del rcds[-1]            # empty string after end of split

        for sr in rcds:
            self.addSample(self.sampleObjType().parseSampleRecordText(sr))
        return self
    #-------------------------

    def write(self, outFile,	# file pathname or open file obj for writing
        writeMeta=True,
        writeHeader=True,
        omitRejects=False,
        ):
        if type(outFile) == type(''): fp = open(outFile, 'w')
        else: fp = outFile

        if writeMeta:
            # make sure we include the actual object type
            self.setMetaItem('sampleObjType', self.sampleObjType.__name__)

            # and the name of the module that type is defined in
            moduleName = inspect.getmodule(self.sampleObjType).__name__
            self.setMetaItem('moduleName', moduleName)
            
            fp.write(self.meta.buildMetaText())

        if writeHeader:	fp.write(self.getHeaderLine() + self.recordEnd)

        for s in self.sampleIterator(omitRejects=omitRejects):
            fp.write(s.getSampleAsText() + self.recordEnd)

        if type(outFile) == type(''): fp.close()      # close if we opened it

        return self
    #-------------------------

    def sampleIterator(self,
        omitRejects=False,
        ):
        for s in self.samples:
            if omitRejects and s.isReject(): continue
            yield s
    #-------------------------

    def addSamples(self, samples,	# list of samples
        ):
        for s in samples:
            self.addSample(s)
        return self
    #-------------------------

    def addSample(self, sample,
        ):
        #if not isinstance(sample, self.sampleObjType):
        if type(sample) != self.sampleObjType:
            raise TypeError('Invalid sample type %s' % str(type(sample)))
        self.samples.append(sample)
        return self
    #-------------------------

    def preprocess(self, preprocessors,  # list of preprocessor (method) names
        ):
        """
        Run the (sample) preprocessors on each sample in the sampleSet.
        Return list of samples that are marked as "isReject" by preprocessors
        """
        if not preprocessors: return []		# no preprocessors to run

        rejects = []

        # save prev sample ID for printing if we get an exception.
        # Gives us a fighting chance of finding the offending record
        prevSampleName = 'very first sample'

        for rcdnum, sample in enumerate(self.sampleIterator()):
            try:
                for pp in preprocessors:
                    sample = getattr(sample, pp)()  # run preproc method 

                if sample.isReject(): rejects.append(sample)

                prevSampleName = sample.getSampleName()
            except:
                sys.stderr.write("\nException in record %s prevID %s\n\n" % \
                                                    (rcdnum, prevSampleName))
                raise
        return rejects
    #-------------------------

    def getSamples(self, omitRejects=False):
        if omitRejects:
            return [s for s in self.sampleIterator(omitRejects=omitRejects) ]
        else:
            return self.samples

    def getSampleIDs(self, omitRejects=False):
        return [s.getID() for s in self.sampleIterator(omitRejects=omitRejects)]

    def getDocuments(self, omitRejects=False):
        return [s.getDocument() for s in \
                                self.sampleIterator(omitRejects=omitRejects)]

    def getNumSamples(self, omitRejects=False):
        return len(self.getSamples(omitRejects=omitRejects))

    def getRecordEnd(self):	return self.recordEnd

    def getSampleObjType(self): return self.sampleObjType
    def getSampleClassNames(self):
        return self.sampleObjType.getClassNames()

    def getY_positive(self):	return self.sampleObjType.getY_positive()
    def getY_negative(self):	return self.sampleObjType.getY_negative()

    def getFieldNames(self):	return self.sampleObjType.getFieldNames()
    def getHeaderLine(self):	return self.sampleObjType.getHeaderLine()

    def setMetaItem(self, key, value):
        if self.meta == None:
            self.meta = SampleSetMetaData()
        self.meta.setMetaItem(key, value)

# end class SampleSet -----------------------------------

class ClassifiedSampleSet (SampleSet):
    """
    IS:     a SampleSet of ClassifiedSamples
    HAS:    list of knownClassnames and Yvalues of the samples, counts, ...
    DOES:   Returns lists of the knownClassNames & Yvalues of the samples.
            Keeps track of counts, get ExtraInfoFieldNames
    """
    def __init__(self, sampleObjType=None):
        super().__init__(sampleObjType=sampleObjType)
        self.numPositives = 0
        self.numNegatives = 0
    #-------------------------

    def addSample(self, sample,		# ClassifiedSample
        ):
        SampleSet.addSample(self, sample)
        if sample.isPositive(): self.numPositives += 1
        else:                   self.numNegatives += 1
        return self
    #-------------------------

    def getKnownClassNames(self, omitRejects=False):
        return [s.getKnownClassName() for s in \
                                self.sampleIterator(omitRejects=omitRejects)]
    def getKnownYvalues(self, omitRejects=False):
        return [s.getKnownYvalue() for s in \
                                self.sampleIterator(omitRejects=omitRejects)]

    def getNumPositives(self):	return self.numPositives
    def getNumNegatives(self):	return self.numNegatives

    def getExtraInfoFieldNames(self):
        return self.sampleObjType.getExtraInfoFieldNames()
# end class ClassifiedSampleSet -----------------------------------

class SampleSetMetaData (object):
    """
    Is:  basically a dictionary of name value pairs that knows how to parse
        and construct a text string representing the items
    """
    metaTag = '#meta '		# start of meta text
    metaEnd = '\n'              # ending of a meta text
    metaSep = '='		# sep between key and value on a meta line

    def __init__(self, ):
        self.metaData = None		# the key:value pairs
    #-------------------------

    def consumeMetaText(self,
                text,	# text that may start with meta info
        ):
        """ if 'text' begins with a meta info, consume it and set the metadata.
            Return the text with the (optional) meta info removed.
        """
        if text.startswith(self.metaTag):
            metaLine, rest = text.split(self.metaEnd, 1)
            self.parseMetaText(metaLine)
            return rest
        else:
            self.metaData = None
            return text
    #-------------------------

    def parseMetaText(self, text,):
        """ Populate the metaData dict
        """
        self.metaData = {}

        parts = text.split()
        for part in parts[1:]:
            # split into name:value pair
            l = part.split(self.metaSep, 1)
            name = l[0]
            if len(l) == 1: value = ''
            else: value = l[1]
            self.metaData[name.strip()] = value.strip()
    #-------------------------

    def buildMetaText(self,):
        """ Return text containing meta info
        """
        text = self.metaTag + " "
        if self.metaData:
            text += " ".join( [ k + self.metaSep + str(v)
                                        for k,v in self.metaData.items() ] )
        return text + self.metaEnd
    #-------------------------

    def setMetaDict(self, d):
        """ set the meta data to the key:value pairs in d"""
        self.metaData = copy(d)

    def getMetaDict(self):
        """ return the meta data as a dict"""
        return copy(self.metaData)

    def setMetaItem(self, key, value):
        if self.metaData == None: self.metaData = {}
        self.metaData[key] = value

    def getMetaItem(self, key):
        return self.metaData.get(key, None)
    #-------------------------

    def hasMetaData(self,):
        return self.metaData != None
    
    def __bool__(self):
        return self.metaData != None

# end class SampleSetMetaData ---------------------

if __name__ == "__main__":
    pass
