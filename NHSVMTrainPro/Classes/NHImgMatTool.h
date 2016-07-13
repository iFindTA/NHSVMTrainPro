//
//  NHImgMatTool.h
//  NHSVMTrainPro
//
//  Created by hu jiaju on 15/10/28.
//  Copyright © 2015年 hu jiaju. All rights reserved.
//

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif
#ifdef __OBJC__
#import <UIKit/UIKit.h>
#import <Foundation/Foundation.h>
#endif

using namespace cv;

@interface NHImgMatTool : NSObject

+ (Mat)cvMatFromUIImage:(UIImage *)image;
+ (UIImage *)UIImageFromCVMat:(Mat)image;

@end
