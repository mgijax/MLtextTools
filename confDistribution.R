# Graphing confidence distributions for predictions

graphConfTrueAndPred = function(filename)
{
    # Graph confidence distributions for known (true) classifications and
    #   predicted classifications
    # Assume file is tab delimited, with header line,
    # Has columns:
    #  ID (string),
    #  True Class (either "yes"/"no" or 1,0),
    #  Pred Class (either "yes"/"no" or 1,0),
    #  Confidence (float),
    #  Abs Value of the confidence (float)
    #  (maybe others not used here)
    #
    # column names: ID, True Class, Pred Class, Confidence, Abs Value
    #  must have these exact names
    #
    # Display 3 distribution histograms in one output window:
    #   distribution of confidences for all predictions
    #   distribution of confidences for true predictions
    #   distribution of confidences for false (FN or FP) predictions
    # All three have the same x axis for comparison (but y axes differ)

    ds = read.table(filename, header=TRUE, sep='\t', row.names="ID")

    # define relevant subsets of the data

    # no idea why I need as.character() on 1 (or both) side of these
	# comparisions. Was getting: "level sets of factors are different"
	# see https://stackoverflow.com/questions/24594981
	# I don't really understand R.

    # True predictions
    truePreds = subset(ds, ds$True.Class == as.character(ds$Pred.Class))
    # False predictions
    FP = subset(ds, ds$True.Class != as.character(ds$Pred.Class) &
		(ds$Pred.Class == "yes" | ds$Pred.Class == 1) )
    FN = subset(ds, ds$True.Class != as.character(ds$Pred.Class) &
		(ds$Pred.Class == "no" | ds$Pred.Class == 0) )

    FPFN = rbind(FP, FN)			# all FP FN together

    # define x axis params
    xMax = max(ds$Abs.Value)
    xTicks = seq(floor(-xMax),ceiling(xMax),0.05) # where tick lines go
    histGroups = 40		# number of histogram boxes

    # define y axis params
    yLwd0 = 1.5		# y-axis line width at x=0
    yMult = 1.3		# y max val multiplier to get y axis to stick up a bit

    # set up graphs (3 per page)
    title=paste("Prediction Confidence Distributions",filename,date(),sep="\n")
    par(mfrow = c(3,1))				# 3 graphs, 1 column

    # for each graph, run hist to get the histogram values (without plotting)
    #   so we can get the yMax for the graph.
    # We use yMax to customize the y axis.
    # Then we run hist again to actually do the plot.

    # plot all predictions
    yMax = max(hist(ds$Confidence,breaks=histGroups,plot=FALSE)$counts)

    hist(ds$Confidence,breaks=histGroups, xlim=c(-xMax,xMax), main=title, 
	    xlab="All Predictions", xaxt='n',ylim=c(0,yMult*yMax), col='yellow')
    axis(1, at=xTicks)
    axis(2, pos=0, col='black', tck=0, lwd=yLwd0, labels=FALSE) # y axis at 0

    # plot true predictions
    yMax = max(hist(truePreds$Confidence,breaks=histGroups,plot=FALSE)$counts)

    hist(truePreds$Confidence,breaks=histGroups, xlim=c(-xMax,xMax), main="",
	    ylim=c(0,yMult*yMax),
		    xlab="True Negative & True Positive Predictions",
		    xaxt='n', col='green')
    axis(1, at=xTicks)
    axis(2, pos=0, col='black', tck=0, lwd=yLwd0, labels=FALSE) # y axis at 0
    
    # plot false predictions
    yMax = max(hist(FPFN$Confidence,breaks=histGroups,plot=FALSE)$counts)

    hist(FPFN$Confidence, breaks=histGroups, xlim=c(-xMax,xMax), main="",
	    ylim=c(0,yMult*yMax),
		    xlab="False Negative and False Positive Predictions",
		    xaxt='n', col='red')
    axis(1, at=xTicks)
    axis(2, pos=0, col='black', tck=0, lwd=yLwd0, labels=FALSE) # y axis at 0
}

graphConfPredictions = function(filename, xmax=0)
{
    # Graph confidence distribution for predicted classifications w/o
    # true classifications (typically because we don't know the true class).
    # Assume file is tab delimited, with header line,
    # Has columns:
    #  ID (string),
    #  Pred Class (either "yes"/"no" or 1/0),
    #  Confidence (float),
    #  Abs Value of the confidence (float)
    #  (maybe others not used here)
    #
    # xmax = the abs value of the biggest x value to graph
    #        ==0 means use the max abs value in the dataset + 0.5 pad.

    ds = read.table(filename, header=TRUE, sep='\t', row.names="ID")

    # x-axis params
    if (xmax==0)xMax = max(ds$Abs.Value) + .5
    else	xMax = xmax
    histGroups = 40		# number of histogram boxes

    # y-axis params
    yLwd0 = 1.5		# y-axis line width at x=0
    yMult = 1.3		# y max val multiplier to get y axis to stick up a bit

    # find yMax from the histogram (w/o plotting)
    yMax = max(hist(ds$Confidence,breaks=histGroups,plot=FALSE)$counts)

    title = paste("Prediction Confidence Distribution",
		    filename,date(),sep="\n")
    hist(ds$Confidence,breaks=histGroups, xlim=c(-xMax, xMax), main=title, 
	xlab="Confidence", ylim=c(0,yMult*yMax),col='blue')
    axis(2, pos=0, col='black', tck=0, lwd=yLwd0, labels=FALSE) # y axis at 0
}
