
#import <UIKit/UIKit.h>
@interface OpenCVWrapper : NSObject

+ (UIImage *)toGray:(UIImage *)source;
+ (UIImage *) rgbImageFromBGRImage: (UIImage *) image;
+ (UIImage *) subRgbImage: (UIImage *) image;
+ (UIImage *)imageWithImage:(UIImage *)image scaledToSize:(CGSize)newSize;
@end
