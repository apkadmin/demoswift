//
//  OpenCVWrapper.m
//  OpenCV
//
//  Created by Dmytro Nasyrov on 5/1/17.
//  Copyright © 2017 Pharos Production Inc. All rights reserved.
//

#ifdef __cplusplus
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"

#import <opencv2/opencv.hpp>
#import "OpenCVWrapper.h"
#import <opencv2/core.hpp>
#import <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/ios.h>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/imgproc/imgproc_c.h>
#import <opencv2/core/types_c.h>

#pragma clang pop
#endif

using namespace std;
using namespace cv;

#pragma mark - Private Declarations

@interface OpenCVWrapper ()

#ifdef __cplusplus

+ (Mat)_grayFrom:(Mat)source;
+ (Mat)_matFrom:(UIImage *)source;
+ (UIImage *)_imageFrom:(Mat)source;

#endif

@end

#pragma mark - OpenCVWrapper

@implementation OpenCVWrapper

#pragma mark Public

+ (UIImage *)toGray:(UIImage *)source {
    cout << "OpenCV: ";
    return [OpenCVWrapper _imageFrom:[OpenCVWrapper _grayFrom:[OpenCVWrapper _matFrom:source]]];
}

#pragma mark Private

+ (Mat)_grayFrom:(Mat)source {
    cout << "-> grayFrom ->";
    
    Mat result;
    cvtColor(source, result, CV_BGR2GRAY);
    
    return result;
}

+ (Mat)_matFrom:(UIImage *)source {
    cout << "matFrom ->";
    
    CGImageRef image = CGImageCreateCopy(source.CGImage);
    CGFloat cols = CGImageGetWidth(image);
    CGFloat rows = CGImageGetHeight(image);
    Mat result(rows, cols, CV_8UC4);
    
    CGBitmapInfo bitmapFlags = kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault;
    size_t bitsPerComponent = 8;
    size_t bytesPerRow = result.step[0];
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image);
    
    CGContextRef context = CGBitmapContextCreate(result.data, cols, rows, bitsPerComponent, bytesPerRow, colorSpace, bitmapFlags);
    CGContextDrawImage(context, CGRectMake(0.0f, 0.0f, cols, rows), image);
    CGContextRelease(context);
    
    return result;
}

+ (UIImage *)_imageFrom:(Mat)source {
    cout << "-> imageFrom\n";
    
    NSData *data = [NSData dataWithBytes:source.data length:source.elemSize() * source.total()];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);

    CGBitmapInfo bitmapFlags = kCGImageAlphaNone | kCGBitmapByteOrderDefault;
    size_t bitsPerComponent = 8;
    size_t bytesPerRow = source.step[0];
    CGColorSpaceRef colorSpace = (source.elemSize() == 1 ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB());
    
    CGImageRef image = CGImageCreate(source.cols, source.rows, bitsPerComponent, bitsPerComponent * source.elemSize(), bytesPerRow, colorSpace, bitmapFlags, provider, NULL, false, kCGRenderingIntentDefault);
    UIImage *result = [UIImage imageWithCGImage:image];
    
    CGImageRelease(image);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return result;
}
+ (UIImage *) rgbImageFromBGRImage: (UIImage *) image {
    // Convert UIImage to cv::Mat
    Mat inputImage;
    UIImageToMat(image, inputImage);
    // If input image has only one channel, then return image.
    if (inputImage.channels() == 1) return image;
    // Convert the default OpenCV's BGR format to RGB.
//    Mat outputImage;
//    cvtColor(inputImage, outputImage, COLOR_RGB2BGR);
    print(inputImage);
    // Convert the BGR OpenCV Mat to UIImage and return it.
    return MatToUIImage(inputImage);
}
+ (UIImage *) subRgbImage: (UIImage *) image {
    // Convert UIImage to cv::Mat
    Mat inputImage;
    UIImageToMat(image, inputImage);
    // If input image has only one channel, then return image.
    if (inputImage.channels() == 1) return image;
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    // Convert the default OpenCV's BGR format to RGB.
    Mat outputImage;
    outputImage.create(128, 128, COLOR_RGB2BGR);
    Mat flat = inputImage.reshape(1, inputImage.total()*inputImage.channels());
    vector<uchar> vec = inputImage.isContinuous()? flat : flat.clone();
    vector<float> copyData;
//    vector<int> vec(inputImage.begin<int>(), inputImage.end<int>());
    for (unsigned i=0; i<vec.size(); ++i)
    copyData.push_back((vec[i] - 127.5)/127.5);
    for (unsigned i=0; i<copyData.size(); ++i)
        cout << copyData[i];
    //    for (int i =0; i < ; i++) {
//
//    }
//    for (int r = 0; r < rows; ++r)
//       {
//           uchar *pOutput = outputImage.ptr<uchar>(r);
//
//           for (int c = 0; c < cols; ++c)
//           {
//               *pOutput = (uchar)vec.at(c);
//               ++pOutput;
//           }
//       }

    return MatToUIImage(inputImage);
}

+ (Mat)_matFromBuffer:(CVImageBufferRef)buffer {

    Mat mat;
    CVPixelBufferLockBaseAddress(buffer,0);
    //Get the data from the first plane (Y)
    void *address = CVPixelBufferGetBaseAddressOfPlane(buffer, 0);
    int bufferWidth = (int)CVPixelBufferGetWidthOfPlane(buffer,0);
    int bufferHeight = (int)CVPixelBufferGetHeightOfPlane(buffer, 0);
    int bytePerRow = (int)CVPixelBufferGetBytesPerRowOfPlane(buffer, 0);
    //Get the pixel format
    OSType pixelFormat = CVPixelBufferGetPixelFormatType(buffer);

    Mat converted;
    //NOTE: CV_8UC3 means unsigned (0-255) 8 bits per pixel, with 3 channels!
    //Check to see if this is the correct pixel format
    if (pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) {
        //We have an ARKIT buffer
        //Get the yPlane (Luma values)
        Mat yPlane = Mat(bufferHeight, bufferWidth, CV_8UC1, address);
        //Get cbcrPlane (Chroma values)
        int cbcrWidth = (int)CVPixelBufferGetWidthOfPlane(buffer,1);
        int cbcrHeight = (int)CVPixelBufferGetHeightOfPlane(buffer, 1);
        void *cbcrAddress = CVPixelBufferGetBaseAddressOfPlane(buffer, 1);
        //Since the CbCr Values are alternating we have 2 channels: Cb and Cr. Thus we need to use CV_8UC2 here.
        Mat cbcrPlane = Mat(cbcrHeight, cbcrWidth, CV_8UC2, cbcrAddress);
        //Split them apart so we can merge them with the luma values
        vector<Mat> cbcrPlanes;
        split(cbcrPlane, cbcrPlanes);

        Mat cbPlane;
        Mat crPlane;
        //Since we have a 4:2:0 format, cb and cr values are only present for each 2x2 luma pixels. Thus we need to enlargen them (by a factor of 2).
        resize(cbcrPlanes[0], cbPlane, yPlane.size(), 0, 0, INTER_NEAREST);
        resize(cbcrPlanes[1], crPlane, yPlane.size(), 0, 0, INTER_NEAREST);

        Mat ycbcr;
        vector<Mat> allPlanes = {yPlane, cbPlane, crPlane};
        merge(allPlanes, ycbcr);
        //ycbcr now contains all three planes. We need to convert it from YCbCr to RGB so OpenCV can work with it
        cvtColor(ycbcr, converted, COLOR_YCrCb2RGB);
    } else {
        //Probably RGB so just use that.
        converted = Mat(bufferHeight, bufferWidth, CV_8UC3, address, bytePerRow).clone();
    }

    //Since we clone the cv::Mat no need to keep the Buffer Locked while we work on it.
    CVPixelBufferUnlockBaseAddress(buffer, 0);

    Mat rotated;
    transpose(converted, rotated);
    flip(rotated,rotated, 1);

    return rotated;
}

//+  resizeImage: (UIImage *) image {
//    InputArray temp = InputArray()
//    temp.
////   return resize( src, <#OutputArray dst#>, Size())
//}

@end
