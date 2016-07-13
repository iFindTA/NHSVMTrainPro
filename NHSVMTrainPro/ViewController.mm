//
//  ViewController.m
//  NHSVMTrainPro
//
//  Created by hu jiaju on 15/10/26.
//  Copyright © 2015年 hu jiaju. All rights reserved.
//

#import "ViewController.h"
#import "svm_train.hpp"
#import "NHImgMatTool.h"

@interface ViewController ()

@property (nonatomic, strong, nullable) UIImageView *imgView;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    CGRect infoRect = CGRectMake(100, 100, 100, 200);
    _imgView = [[UIImageView alloc] initWithFrame:infoRect];
    _imgView.contentMode = UIViewContentModeScaleAspectFit;
    [self.view addSubview:_imgView];
    
    NSString *document = NSTemporaryDirectory();
    NSLog(@"temp path :%@",document);
    
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSString *filePath = [[NSBundle mainBundle] pathForResource:@"Trains" ofType:@"bundle"];
    NSURL *bundleURL = [NSURL fileURLWithPath:filePath];
    NSArray *contents = [fileManager contentsOfDirectoryAtURL:bundleURL
                                   includingPropertiesForKeys:@[]
                                                      options:NSDirectoryEnumerationSkipsHiddenFiles
                                                        error:nil];
    
    vector< vector<string> > paths;
    for (NSURL *tmpPath in contents) {
        NSArray *datas = [fileManager contentsOfDirectoryAtURL:tmpPath
                                    includingPropertiesForKeys:@[]
                                                       options:NSDirectoryEnumerationSkipsHiddenFiles
                                                         error:nil];
        vector<string> tmpPaths;
        for (NSURL *name in datas) {
            NSString *filePath = [name absoluteString];
            if ([filePath rangeOfString:@"file://"].location != NSNotFound) {
                filePath = [filePath stringByReplacingOccurrencesOfString:@"file://" withString:@""];
            }
            string cPath = [filePath UTF8String];
            tmpPaths.push_back(cPath);
        }
        paths.push_back(tmpPaths);
    }
    
    /// train svm 
    svm_trainForPath(paths);
    
    int count = 10;
    int numPerLine = 6;
    CGSize mainSize = [UIScreen mainScreen].bounds.size;
    CGFloat distance = 24;
    CGFloat itemW = (mainSize.width-(numPerLine+1)*distance)/numPerLine;
    int index = 20;///0-50
    for (int i = 0;i < count ;i++) {
        int row = i / numPerLine;
        int col = i % numPerLine;
        string first = paths[i][index];
        Mat img = imread(first.c_str(), 0);
        float ret = svm_recog(img);
        cout << "result : " << ret << endl;
        UIImage *image = [NHImgMatTool UIImageFromCVMat:img];
        infoRect = CGRectMake(distance+(itemW+distance)*col, 100+(distance+itemW)*row, itemW, itemW);
        UIImageView *imageView = [[UIImageView alloc] initWithFrame:infoRect];
        imageView.contentMode = UIViewContentModeScaleAspectFit;
        imageView.image = image;
        [self.view addSubview:imageView];
    }
    
    NSString *testPath = [[NSBundle mainBundle] pathForResource:@"test" ofType:@"jpg"];
    if ([testPath rangeOfString:@"file://"].location != NSNotFound) {
        testPath = [testPath stringByReplacingOccurrencesOfString:@"file://" withString:@""];
    }
    string cPath = [testPath UTF8String];
    Mat img = imread(cPath.c_str(), 0);
    float ret = svm_recog(img);
    cout << "test result : " << ret << endl;
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
