# Graphing confidence distributions for predictions


graphRefSectionLengthDistribution = function(filename, xmax=0)
{
    # Graph dist of lengths of the reference sections we've found
    #   by searching for certain keywords
    # Assume file is | delimited, with header line,
    # Needs columns:   ID and Percent.after

    ds = read.table(filename, header=TRUE, sep='|', row.names="ID")

    # x-axis params
    if (xmax==0)xMax = max(ds$Percent.after) + .5
    else	xMax = xmax
    histGroups = 80		# number of histogram boxes

    # y-axis params
    yLwd0 = 1.5		# y-axis line width at x=0
    yMult = 1.3		# y max val multiplier to get y axis to stick up a bit

    # find yMax from the histogram (w/o plotting)
    yMax = max(hist(ds$Percent.after,breaks=histGroups,plot=FALSE)$counts)

    title = paste("Reference Section Length Distribution",
	filename,date(),sep="\n")
    xlabel = paste("Percent of Doc After Ref Keyword\nStd dev = ",
	sd(ds$Percent.after), sep=' ')
    hist(ds$Percent.after,breaks=histGroups, xlim=c(0, xMax), main=title, 
	xlab=xlabel,
	ylim=c(0,yMult*yMax),col='blue')
    # axis(2, pos=0, col='black', tck=0, lwd=yLwd0, labels=FALSE) # y axis at 0
}
graphNumMice = function(filename, xmax=0)
{
    # Assume file is | delimited, with header line,
    # Needs columns:   ID and Num.mice.before

    ds = read.table(filename, header=TRUE, sep='|', row.names="ID")

    # x-axis params
    if (xmax==0)xMax = max(ds$Num.mice.before) + .5
    else	xMax = xmax
    histGroups = 80		# number of histogram boxes

    # y-axis params
    yLwd0 = 1.5		# y-axis line width at x=0
    yMult = 1.3		# y max val multiplier to get y axis to stick up a bit

    # find yMax from the histogram (w/o plotting)
    yMax = max(hist(ds$Num.mice.before,breaks=histGroups,plot=FALSE)$counts)

    title = paste("Number of 'mice' usage Distribution",
	filename,date(),sep="\n")
    xlabel = paste("Mice References\nStd dev = ",
	sd(ds$Num.mice.before), sep=' ')
    hist(ds$Num.mice.before,breaks=histGroups, xlim=c(0, xMax), main=title, 
	xlab=xlabel,
	ylim=c(0,yMult*yMax),col='blue')
    # axis(2, pos=0, col='black', tck=0, lwd=yLwd0, labels=FALSE) # y axis at 0
}
