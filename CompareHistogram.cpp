#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

# define HISTSIZE 128
/******** Constant variables **************/

// the number of bins
int histSize[] = {HISTSIZE}; // (256, 128)
// grayscale ranges from 0 to 255
float g_ranges[] = { 0.0, 255.0 };
const float* ranges[] = { g_ranges };
// use the only channel available 
int channels[] = {0};
// the number of images used to obtain the histogram pattern
int num_of_images_for_histogram_pattern = 10;
// the starting bin of the histogram pattern
int start_bin = 0; // (10, 21)
// the end bin of the histogram pattern
int end_bin = 24; // (21, 43)
// the number of frames to be blended
int num_of_frames_blended = 25;
// the number of runs to find the correlataion value 
// between the pattern and a small part of an image
int num_of_correlation_runs = 1; // (22,44)
// the histogram pattern matrix 
Mat hist_pattern(end_bin - start_bin, 1, CV_32FC1); // (22, 11)
//int range = end_bin - start_bin;
// number of images blended in real time
int num_of_frames_blended_in_real_time = 5;


/************************* All helper functions *****************************/
// build histogram pattern from num_of_images_for_histogram_pattern images
// each of them is blended from num_of_frames_blended frames 
// and save the pattern to hist_test
// INPUT: an array of matrix data source of images
// OUTPUT: none
void buildPatternHistAverage(Mat frames[])
{
	// run through all images to determine the histogram pattern
	for(int k = 0; k < num_of_images_for_histogram_pattern; k++)
	{				
		// add values of each bin from the histogram to the corresponding bin 
		for (int i = start_bin; i < end_bin; i++)  
		{			
			// if this is the first image, then just add its bin value
			if (k==0)
				{hist_pattern.at<float>(i - start_bin ,0)= frames[k].at<float>(i,0);} 
			// if not the first image, then add its bin value to the current value of the pattern's corresponding bin
			else
				{hist_pattern.at<float>(i - start_bin ,0) = hist_pattern.at<float>(i - start_bin ,0) + frames[k].at<float>(i,0);} 
		}		
	}

	// now divide each bin value of the pattern by the number of images used
	// to obtain the average histogram pattern
	for (int i = start_bin; i < end_bin; i++) 
	{
		hist_pattern.at<float>(i - start_bin, 0) = hist_pattern.at<float>(i - start_bin, 0) / (num_of_images_for_histogram_pattern);
	}
}

// compute the normalized gray-scale histogram from an image's matrix data source
// INPUT: a matrix source of an image
// OUTPUT: a matrix contains normalized gray-scale histogram from the original source
Mat computeGrayHistogram(Mat src)
{
	// initialization
	Mat gray, hist;
	// convert the image to gray-scale
	cvtColor( src, gray, CV_BGR2GRAY );		
	// calculate the histograms for the gray-scale image and store the histogram in hist
	calcHist( &gray, 1, channels, Mat(), hist, 1, histSize, ranges, true, false );
	// normalize the histogram, so that each bin value is between 0 and 1
	normalize( hist, hist, 0, 1, NORM_MINMAX, -1, Mat() );	
	// return the normalized gray-scale histogram from the original image source
	return hist;		
}


// compute the correlation value between the histogram pattern 
// and a small fraction of a gray-scale image
// INPUT: a matrix source of a gray-scale image's histogram
// OUTPUT: a double correlation value shows how well the match between 
//		   the pattern and the image
double patternHistMatch(Mat hist)
{	
	// initialize the maximum correlation value
	double max = 0;
	// initialize a matrix to hold bin values of a small part of the image
	Mat hist_sample(end_bin - start_bin, 1, CV_32FC1); 
	// loop through to find the best match between the pattern and a small part of the image
	for (int i=0; i < num_of_correlation_runs; i++)			 
	{
		// build a matrix which has the same length as the histogram pattern has
		for (int j = 0; j < end_bin - start_bin; j++)         // (22, 11)
		{
			hist_sample.at<float>(j,0)= hist.at<float>(i+j,0);
		}		
		// use the first correlation method of the openCV compareHist() method
		int compare_method = 0; 				
		// obtain the correlation value
		double value = compareHist( hist_pattern, hist_sample, compare_method );
		// pay attention only to the maximum correlation value for each run
		if (value > max)
		{
			max = value;
		}				
	}
	// return the final maximum correlation value after all runs
	return max;
}

// Computes the 1D histogram and returns an image of it.
// INPUT: a gray-scale histogram's matrix data
// OUTPUT: a matrix data of the original image's 1D histogram
Mat getHistogramImage(const Mat &hist)
{
	// Get min and max bin values
	double maxVal=0;
	double minVal=0;
	minMaxLoc(hist, &minVal, &maxVal, 0, 0);
	
	// Image on which to display histogram
	Mat histImg(histSize[0], histSize[0],CV_8U,Scalar(255));
	
	// set highest point at 90% of nbins
	int hpt = static_cast<int>(0.9*histSize[0]);
	
	// Draw a vertical line for each bin
	for( int h = 0; h < histSize[0]; h++ ) 
	{
		float binVal = hist.at<float>(h);
		int intensity = static_cast<int>(binVal*hpt/maxVal);
		// This function draws a line between 2 points
		line(histImg,Point(h,histSize[0]),
			Point(h,histSize[0]-intensity),
			Scalar::all(0));
	}
	return histImg;
}


// save 10 grayscale histograms of 10 different images
// each of them is made up of 25 distinct frames extracted from video feed
// Input: an array of type Mat to stores 10 histograms
// Output: none
void buildSrcImgForPattern(Mat frames[])
{
	// initialization
	Mat frame; // hold an individual frame's data set
	Mat dest;  // hold data set obtained by blending two frames	
	double alpha = 0.5; double beta = 0.5; // constant varialbe used for blending task
	int i=1; // keep track of blending 25 frames
	int j=0; // keep track of how many steps needed to reach 10 images in which each image i
	int k=0; // keep track of how many items in the frames array
	int timeWait = 40; // how many miliseconds apart between reading two frames
	int timeInterval = 300; // waittime before repeating the task of blending 25 frames

	// set properties for the live video feed
	VideoCapture capture = VideoCapture(0); // '0' means to receive input from a webcam
	namedWindow("webcam", CV_WINDOW_AUTOSIZE); 
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 320); // set the width of the video frame
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);// set the height of the video frame

	// prepare the very first frame for blending task
	capture.read(dest); // read a frame from live video feed
	dest = computeGrayHistogram(dest); // compute grayscale histogram of that frame
	waitKey(timeWait); // wait for a certain time before reading a new frame
	
	while(j < num_of_images_for_histogram_pattern * num_of_frames_blended + 1)
	{		
		if (i > num_of_frames_blended) 
		{	
			// reset parameters and wait 
			i = 1;			
			frames[k] = dest; // save the data into frames array
			k = k + 1;  
			waitKey(timeInterval);
			capture.read(dest);
			dest = computeGrayHistogram(dest);
			waitKey(timeWait);
		}
		capture.read(frame);
		imshow("webcam", frame);
		// function to blend two grayscale histograms
		addWeighted( dest, alpha, computeGrayHistogram(frame), beta, 0.0, dest);;		
		i = i +1;
		j = j + 1;
		waitKey(timeWait);
	}								
}


// calculate how rich the black part is in a histogram
// by summing up all bins value from start_bin to end_bin
// and divide by the number of bins in that range
// help in the case a bright image has a high correlation value
// INPUT: histogram data of an image
// OUTPUT: a number between 0-1 
double computeAreaUnderHistogram(Mat hist)
{
	double avg_sum = 0;
	for (int i = start_bin; i< end_bin; i++)
	{
		avg_sum = avg_sum + hist.at<float>(i, 0);
	}
	return avg_sum / (end_bin - start_bin);
}

// write pattern to a text file
void writePatternToFile()
{	
	Mat frames[50]; // any number greater or equal to num_of_images_for_histogram_pattern
	buildSrcImgForPattern(frames);
	
	buildPatternHistAverage(frames);
	
	// open a file 
	ofstream output;
	output.open("128bins.txt");
	for (int i=0; i < num_of_images_for_histogram_pattern; i++)
	{
		printf("%f \n", patternHistMatch(frames[i]));
		output<<"Image "<<i+1<<": \n";
		for (int j=0;j < HISTSIZE; j++)
		{
			output<<frames[i].at<float>(j,0)<<"\n";
		}
		output<<"\n\n";
	}
	output.close();

	// only for pattern
	output.open("pattern_new.txt");
	//output<<"\n\n"<<"Pattern: \n";
	for (int i = start_bin; i < end_bin; i++) 
	{
		output<<hist_pattern.at<float>(i - start_bin, 0)<<"\n";
	}

	output.close();
	
}

// read pattern stored in a text file
// INPUT: none
// OUTPUT: none
void readPatternFromFile()
{
	char str [20];
	int k = 0;
	int const range = end_bin - start_bin + 1;
	double patternArray[51]; // must be greater or equal to range
	string line;
	ifstream myfile ("pattern_new.txt");
	if (myfile.is_open())
	{
		while ( myfile.good() )
		{
			getline (myfile,line);			
			for (int i = 0; i<line.size(); i++)
			{	
				str[i] = line[i];
			}
			patternArray[k] = atof(str);
			k++;
		}
		myfile.close();
	}
	else cout << "Unable to open file"; 
	for (int i = 0; i<end_bin-start_bin; i++)
	{
		hist_pattern.at<float>(i, 0) = patternArray[i];		
	}
}

// real-time running to detect pot
// INPUT: none
// OUTPUT: none
void realTimeRunning()
{
	// Initialization
	VideoCapture capture = VideoCapture(0);
	namedWindow("webcam", CV_WINDOW_AUTOSIZE);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

	Mat frame;
	Mat dest;	
	capture.read(dest);
	dest = computeGrayHistogram(dest);
	double alpha = 0.5; double beta = 0.5;
	int timeWait = 40; // how many miliseconds apart between reading two frames
	int timeInterval = 300; // waittime before repeating the task of blending num_of_frames_blended_in_real_time frames

	capture.read(dest);
	dest = computeGrayHistogram(dest);
	waitKey(timeWait);
	int i=1;
	int count = 0;
	float corr_threshold = 0.6; // found by observation
	float area_threshold = 0.2; // found by observation
	int num_of_times_running_in_real_time = 200; // how many times we want the algorithm to run in real time
	float area_value = 0.0; // holds the returned area under the curve value
	float corr_value = 0.0; // holds the returned correlation value

	while(count < num_of_frames_blended_in_real_time * num_of_times_running_in_real_time)	
	{		
		if (i > num_of_frames_blended_in_real_time)
		{
			i = 1;
			corr_value = patternHistMatch(dest);
			area_value = computeAreaUnderHistogram(dest);
			printf("Correlation value: %f \n", corr_value);					
			printf("Area under curve : %f \n", area_value);
			if (corr_value >= corr_threshold)
			{
				if (area_value >= area_threshold)
				{
					printf("HAS POT \n");
				}
				else
				{
					printf("NO POT \n");
				}
			}
			else
			{
				printf("NO POT \n");
			}
			waitKey(timeInterval);
			capture.read(dest);
			dest = computeGrayHistogram(dest);
			waitKey(timeWait);
		}
		// blending images
		capture.read(frame);
		imshow("webcam", frame);				
		addWeighted( dest, alpha, computeGrayHistogram(frame), beta, 0.0, dest);;
		// updating variables 				
		i = i +1;
		count++;
		waitKey(timeWait);
	}	
}

void main()
{
	//writePatternToFile();
	readPatternFromFile();
	realTimeRunning();	
}


