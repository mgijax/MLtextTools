Analysis (and ultimate python module) for detecting/removing reference sections from text extracted from PDFs

July 13, 2018
Did 3 experiments, comparing against Feb 13 approach.

Feb 13:
    search backward for all the words, w/ max fraction of ref section < .4
    of doc

    Curators evaluated 500 predicted reference sections using this algorithm.
    These are here: 
    https://docs.google.com/spreadsheets/d/1v9aILIK47yCxe7dtwuZBt1Vw7EFxVFKYfBP4lHKlHKc/edit#gid=2003513120
    (I made a few corrections to these evaluations in July as I discovered them)

Jul 12:
    search forward, differentiating "references", "literature cited", and
    "reference" as primary key terms, and searching for them 1st. If found,
    go with that. If not found, search for secondary terms: "acknowledgements",
    "conflict of interest" (continue < .4)

    This fixed a common issue of finding "Acknowledgements" after "References"
    which addressed a fair number of predicted reference sections being found
    "too late"

Jul 13:
    search foward, but this time add "reference" as secondary (since it often
    appears in figures/tables, both before true reference section or after).
    Add "min fraction", meaning the ref section must be > min fraction (5%)
    of doc (to skip summary of "references" that sometimes occur at the
    end of some docs)

Jul 13.2:
    like Jul 13, but search backward,

Compared each of these "july" approaches to the originally evaluated Feb 13 set.
Jul 13.2 is probably the one to go with, but it is very close to the other two.

The major improvement of all of them is to stop finding "Acknowledgements" or
"Conflict of Interest" after the refs section and calling them the refs section.

I manually spot checked some of the original ones that Feb 13 predicted as
"too late" which the new algorithms predicted "earlier" (69 of these in
Jul 3.2). With few exceptions, these correctly found the reference section
(most are from Sci Rep and Nat Commun and the docs I looked at all followed
the same pattern)

So if we count the 356 manually evalulated as "Good" (as of Feb 13) that the
new algorithm predicts "same" and the "Too Late" that are now predicted
"earlier" as all accurate predictions,

We get:
425/500 (85%) predicted correctly 
47/500 (9%) have no prediction
25/500 (5%) have incorrect predictions

Of these 5% incorrect predictions:
18/500 (4%) are "too late" meaning nothing important in the doc would be
		skipped by omitting the predicted section
7/500 (1%) are "too early" meaning something important may be skipped

Caveat: the randomly selected set of predictions that curators evaluated did
not include references from all journals.

I also compared July 12 to July 13.2 (the full set of predictions, not just
the evaluated set):
30,562 predictions
29,801 (98%) are predicted the same by the two algorithms

I spot checked some of the differences. In no case I checked did July 12 do
a better job. In most cases the differences didn't matter much (e.g.,
erroneously calling the Ref section at the very end of the doc vs. not
predicting a ref section) or there is really nothing easy to be done, e.g., 
an additional reference section in the supp data and the supp data is so big
it pushes the real refs section > 40% into the document.
