//  COLOR MAG
//  OpenCVWrapper.mm
//  FilterOpenCV
//
//  Created by Brandon Nghe on 3/1/19.
//  Copyright © 2019 Brandon Nghe. All rights reserved.
// test2
// test3


#import <vector>
#import <opencv2/opencv.hpp>
#import <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#import <Foundation/Foundation.h>
#import "OpenCVWrapper.h"

using namespace cv;
using namespace std;

double cutoff_freq_low = 0.05; //Hz, 24BPM
double cutoff_freq_high = 0.4; //Hz 240BPM
double lambda_c = 16;
double alpha = 100;
double chrom_attenuation = 0.1;
double exaggeration_factor = 2.0;
double delta = 0;
double lambda = 0;
int input_fps = 30;
int currentFrame = 0;
/*
bool first_frame_done = false;
int lap_pyramid_levels = 9;

std::vector< cv::Mat > img_vec_gaus_pyramid_;
std::vector< cv::Mat > img_vec_lowpass_1_;
std::vector< cv::Mat > img_vec_lowpass_2_;
std::vector< cv::Mat > img_vec_filtered_;

cv::Mat img_spatial_filter_;
cv::Mat img_motion_;
cv::Mat img_motion_mag_;

 */

/// Converts an UIImage to Mat.
/// Orientation of UIImage will be lost.
static void UIImageToMat(UIImage *image, cv::Mat &mat) {
    assert(image.size.width > 0 && image.size.height);
    assert(image.CGImage != nil || image.CIImage != nil);

    // Create a pixel buffer.
    NSInteger width = image.size.width;
    NSInteger height = image.size.height;
    cv::Mat mat8uc4 = cv::Mat((int)height, (int)width, CV_8UC4);

    // Draw all pixels to the buffer.
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    if (image.CGImage) {
        // Render with using Core Graphics.
        CGContextRef contextRef = CGBitmapContextCreate(mat8uc4.data, mat8uc4.cols, mat8uc4.rows, 8, mat8uc4.step, colorSpace, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrderDefault);
        CGContextDrawImage(contextRef, CGRectMake(0, 0, width, height), image.CGImage);
        CGContextRelease(contextRef);
    } else {
        // Render with using Core Image.
        static CIContext* context = nil; // I do not like this declaration contains 'static'. But it is for performance.
        if (!context) {
            context = [CIContext contextWithOptions:@{ kCIContextUseSoftwareRenderer: @NO }];
        }
        CGRect bounds = CGRectMake(0, 0, width, height);
        [context render:image.CIImage toBitmap:mat8uc4.data rowBytes:mat8uc4.step bounds:bounds format:kCIFormatRGBA8 colorSpace:colorSpace];
    }
    CGColorSpaceRelease(colorSpace);

    // Adjust byte order of pixel.
    cv::Mat mat8uc3 = cv::Mat((int)width, (int)height, CV_8UC3);
    cv::cvtColor(mat8uc4, mat8uc3, COLOR_RGBA2BGR);

    mat = mat8uc3;
}

/// Converts a Mat to UIImage.
static UIImage *MatToUIImage(cv::Mat &mat) {

    // Create a pixel buffer.
    //printf("%zu\n", mat.elemSize());
    assert(mat.elemSize() == 1 || mat.elemSize() == 3);
    cv::Mat matrgb;
    if (mat.elemSize() == 1) {
        cv::cvtColor(mat, matrgb, COLOR_GRAY2RGB);
    } else if (mat.elemSize() == 3) {
        cv::cvtColor(mat, matrgb, COLOR_BGR2RGB);
    }

    // Change a image format.
    NSData *data = [NSData dataWithBytes:matrgb.data length:(matrgb.elemSize() * matrgb.total())];
    CGColorSpaceRef colorSpace;
    if (matrgb.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef imageRef = CGImageCreate(matrgb.cols, matrgb.rows, 8, 8 * matrgb.elemSize(), matrgb.step.p[0], colorSpace, kCGImageAlphaNone|kCGBitmapByteOrderDefault, provider, NULL, false, kCGRenderingIntentDefault);
    UIImage *image = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);

    return image;
}

/// Restore the orientation to image.
static UIImage *RestoreUIImageOrientation(UIImage *processed, UIImage *original) {
    if (processed.imageOrientation == original.imageOrientation) {
        return processed;
    }
    return [UIImage imageWithCGImage:processed.CGImage scale:1.0 orientation:original.imageOrientation];
}

#pragma mark -

void init_src(cv::Mat &src){
    // Crops to center 512 x 512 of Input
    src = src(cv::Rect(244,14,512,512));
    src.convertTo(src, CV_32FC3, 1.0/255.0f);
    cvtColor(src, src, COLOR_BGR2Lab);
}

void outputimage(cv::Mat &src){
    // Converts back to BGR and CV_8UC4
    cvtColor(src, src, COLOR_Lab2BGR);
    src.convertTo(src, CV_8UC4, 255.0, 1.0/255.0);
}

void buildGaussPyrFromImg(const Mat &img, const int levels, vector<Mat> &pyr)
{
    pyr.clear();
    cv::Mat currentLevel = img;

    for (int level = 0; level < levels; ++level) {
        Mat down;
        pyrDown(currentLevel, down);
        pyr.push_back(down);
        currentLevel = down;
    }
}

int getOptimalBufferSize(int fps){
    // Calculate number of images needed to represent 2 seconds of film material
    unsigned int round = (unsigned int) std::max(2*fps,16);
    // Round to nearest higher power of 2
    round--;
    round |= round >> 1;
    round |= round >> 2;
    round |= round >> 4;
    round |= round >> 8;
    round |= round >> 16;
    round++;

    return round;
}


void concat(const Mat &frame, Mat &dst, int maxImages)
{
    //Reshaped in 1 column
    Mat reshaped = frame.reshape(frame.channels(), frame.cols*frame.rows). clone();

    reshaped.convertTo(reshaped, CV_32FC3);

    //First Frame
    if (dst.cols == 0) {
        reshaped.copyTo(dst);
    }
    //Later Frames
    else {
        hconcat(dst, reshaped, dst);
    }

    // If dst reaches maximum, delete the first column
    if (dst.cols > maxImages && maxImages != 0){
        dst.colRange(1, dst.cols).copyTo(dst);

    }

    printf("%zu\n", dst.cols);
}

void createIdealBandpassFilter(Mat &filter, double cutoffLo, double cutoffHi, double framerate)
{
    int width = filter.cols;
    int height = filter.rows;

    // Calculate frequencies according to framerate and size
    float fl = 2 * cutoffLo * width / framerate;
    float fh = 2 * cutoffHi * width / framerate;

    // Create the filtermask, looks like the quarter of a circle
    for(int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if(x >= fl && x <= fh)
                filter.at<float>(y,x) = 1.0f;
            else
                filter.at<float>(y,x) = 0.0f;
        }
    }
}

void idealFilter(const Mat &src, Mat &dst , double cutoffLo, double cutoffHi, double framerate)
{
    Mat channels[3];
    split(src, channels);

    int width = getOptimalDFTSize(src.cols);
    int height = getOptimalDFTSize(src.rows);

    // Apply filter on each channel individually
    for (int curChannel = 0; curChannel < 3; ++curChannel) {
        Mat current = channels[curChannel];
        Mat tempImg;


        copyMakeBorder(current, tempImg,
                       0, height - current.rows,
                       0, width - current.cols,
                       BORDER_CONSTANT, Scalar::all(0));

        // DFT
        dft(tempImg, tempImg, DFT_ROWS | DFT_SCALE);

        // construct Filter
        Mat filter = tempImg.clone();
        createIdealBandpassFilter(filter, cutoffLo, cutoffHi, framerate);

        // apply
        mulSpectrums(tempImg, filter, tempImg, DFT_ROWS);

        // inverse
        idft(tempImg, tempImg, DFT_ROWS | DFT_SCALE);

        tempImg(cv::Rect(0, 0, current.cols, current.rows)).copyTo(channels[curChannel]);
    }
    merge(channels, 3, dst);
    normalize(dst, dst, 0, 1, NORM_MINMAX);
}

void deConcat(const Mat &src, int position, const cv::Size &frameSize, Mat &frame){
    printf("%zu\n", src.cols);
    Mat line = src.col(position).clone();
    frame = line.reshape(3, frameSize.height).clone();
}

void amplify(const Mat &src, Mat &dst){
    dst = src * alpha;
}

void buildImgFromGaussPyr(const Mat &pyr, const int levels, Mat &dst, cv::Size size)
{
    printf("\nERROR b1");
    Mat currentLevel = pyr.clone();

    printf("\nERROR b2");

    for (int level = 0; level < levels; ++level) {
        printf("\nERROR b3");
        Mat up;
        printf("\nERROR b4");
        pyrUp(currentLevel, up);
        printf("\nERROR b5");
        currentLevel = up;
    }
    // Resize the image to comprehend errors due to rounding
    resize(currentLevel,currentLevel,size);
    currentLevel.copyTo(dst);
}

@implementation OpenCVWrapper

+ (nonnull UIImage *)cvtColorMagnify:(nonnull UIImage *)image {
    int levels = 9;
    cv::Mat src, output, color, filteredFrame, filteredMat, downSampledFrame, downSampledMat;
    std::vector<cv::Mat> inputFrames, inputPyramid, filteredFrames;

    int offset = 0;

    // Convert Input to Lab Color Space
    UIImageToMat(image, src);
    init_src(src);

    // Save input frame
    inputFrames.push_back(src);

    // 1. Spatial Filter, GAUSSIAN
    buildGaussPyrFromImg(src, levels, inputPyramid);

    // 2. Concatentate the smallest frame from pyramid
    downSampledFrame = inputPyramid.at(levels-1);
    concat(downSampledFrame, downSampledMat, getOptimalBufferSize(input_fps));

    ++currentFrame;
    ++offset;

    printf("\nERROR 2");

    // 3. Temporal Filter
    idealFilter(downSampledMat, filteredMat, cutoff_freq_low, cutoff_freq_high, input_fps);

    printf("\nERROR 3");

    // 4. Amplify Color Motion
    amplify (filteredMat, filteredMat);

    printf("\nERROR 4");

    for (int i = 0; i < 63; i++){

        // 5. Deconcat
        printf("\nERROR 4.5");
        printf("%zu\n", downSampledFrame.cols);
        deConcat(filteredMat, i, downSampledFrame.size(), filteredFrame);

        printf("\nERROR 5");

        // 6. Reconstruct Color Image from Gauss Pyramid
        buildImgFromGaussPyr(filteredFrame, levels, color, src.size());

        printf("\nERROR 6");

        // 7. Add Color to Original Image
        output = inputFrames.front()+color;

        printf("\nERROR 7");
        double min, max;
        minMaxLoc(output, &min, &max);
    }

    printf("%zu\n", output.empty());
    Mat out2;
    printf("\nERROR 8");
    if (output.empty() == 1){
        printf("\nSRC");
        out2 = src;
    } else {
        printf("\nOUT");
        out2 = output;
    }
    printf("\nERROR 9");
    outputimage(out2);
    UIImage *filteredImage = MatToUIImage(out2);
    return RestoreUIImageOrientation(filteredImage, image);
}

@end
