//
//  ViewController.swift
//  movanai
//
//  Created by Nguyen Van An on 4/4/21.
//

import UIKit
import AVFoundation
import Vision
import CoreML
import TensorFlowLite
import TensorFlowLiteC

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate{

    @IBOutlet weak var containerView: UIView!
    
    let imagePreview = UIImageView()
    private let captureSession = AVCaptureSession()
    private lazy var previewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private var modelDataHandler: ModelDataHandler? = ModelDataHandler(modelFileInfo: TFModel.faceModel)
    private var faceDetection = FaceDetection()
    var optionsFace: OptionsFace = OptionsFace(
        numClasses: 1,
        numBoxes: 896,
        numCoords: 16,
        keypointCoordOffset: 4,
        ignoreClasses: [],
        scoreClippingThresh: 100.0,
        minScoreThresh: 0.75,
        numKeypoints: 6,
        numValuesPerKeypoint: 2,
        boxCoordOffset: 0,
        xScale: 128,
        yScale: 128,
        wScale: 128,
        hScale: 128,
        applyExponentialOnBoxSize: false,
        reverseOutputOrder: true,
        sigmoidScore: true, flipVertically: false)
    
    var anchors = AnchorOption(
        inputSizeWidth: 128, inputSizeHeight: 128,
        minScale: 0.1484375,
        maxScale: 0.75,
        anchorOffsetX: 0.5,
        anchorOffsetY: 0.5,
        numLayers: 4,
        featureMapWidth: [], featureMapHeight: [],
        strides: [8, 16, 16, 16],
        aspectRatios: [1.0],
        reduceBoxesInLowestLayer: false,
        interpolatedScaleAspectRatio: 1.0,
        fixedAnchorSize: true)
    var _normalizeInput: NormalizeOp = NormalizeOp(127.5, 127.5)
    var _anchors: [Anchor] = []
    let view1 = UIView()
   
    override func viewDidLoad() {
        super.viewDidLoad()
        self.addCameraInput()
        self.showCameraFeed()
        self.getCameraFrames()
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
        imagePreview.frame = CGRect(x: 10, y: 10, width: 128, height: 128)
        self.view.addSubview(imagePreview)
        self.view.bringSubviewToFront(imagePreview)
        view1.frame = CGRect(x: 0, y: 0, width: 10, height: 10)
        view1.backgroundColor = UIColor.red
        imagePreview.addSubview(view1)
//            load()
        
    }
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
      
    }
    
    func convertCVPixelToUIImage(pixelBuffer: CVPixelBuffer) -> UIImage {
        let ciimage : CIImage = CIImage(cvPixelBuffer: pixelBuffer)
        let imageView : UIImage = self.convertCIToUIImage(cmage: ciimage)
        return imageView
    }
    
    func convertCIToUIImage(cmage: CIImage) -> UIImage {
         let context:CIContext = CIContext.init(options: nil)
         let cgImage:CGImage = context.createCGImage(cmage, from: cmage.extent)!
         let image:UIImage = UIImage.init(cgImage: cgImage)
         return image
    }

    private func addCameraInput() {
        self.captureSession.sessionPreset = .medium
        if #available(iOS 11.1, *) {
            guard let device = AVCaptureDevice.DiscoverySession(
                    deviceTypes: [.builtInWideAngleCamera, .builtInDualCamera, .builtInTrueDepthCamera],
                    mediaType: .video,
                    position: .front).devices.first else {
                fatalError("No back camera device found, please make sure to run SimpleLaneDetection in an iOS device and not a simulator")
            }
            let cameraInput = try! AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(cameraInput) {
                self.captureSession.addInput(cameraInput)
            }
        }
    }
    private func showCameraFeed() {
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.layer.bounds
        previewLayer.videoGravity = .resizeAspectFill
        containerView.layer.addSublayer(previewLayer)
        captureSession.startRunning()
    }
    private func getCameraFrames() {
        self.videoDataOutput.videoSettings = [(kCVPixelBufferPixelFormatTypeKey as NSString) : NSNumber(value: kCVPixelFormatType_32BGRA)] as [String : Any]
        self.videoDataOutput.alwaysDiscardsLateVideoFrames = true
        self.videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera_frame_processing_queue"))
        self.captureSession.addOutput(self.videoDataOutput)
        guard let connection = self.videoDataOutput.connection(with: AVMediaType.video),
              connection.isVideoOrientationSupported else { return }
        connection.videoOrientation = .portrait
    }
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        guard let frame = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            debugPrint("unable to get image from sample buffer")
            return
        }
        let img = convertCVPixelToUIImage(pixelBuffer: frame)
        let bckgd = ResizeImage(image: img, targetSize: CGSize(width: 128, height: 128))
//        load()
        let resultModel = modelDataHandler?.runModel(onFrame: buffer(from: bckgd)!)
          if let result = resultModel {
              let anchorsData = faceDetection.getAnchors(options: anchors)
              let a = Array(result)[0].value
              let b = Array(result)[1].value
         
              var detection: [Detection] = []
              if a.count == 14336 {
                  detection = faceDetection.process(options: optionsFace, rawScores: b, rawBoxes: a, anchors: anchorsData)
              } else {
                  detection = faceDetection.process(options: optionsFace, rawScores: a, rawBoxes: b, anchors: anchorsData)
              }
 let detectionFace = faceDetection.origNms(detections: detection, threshold: 0.3, img_width: 128, img_height: 128)
            DispatchQueue.main.async {
                for i in 0..<detectionFace.count {
//                 print("CVPixelBufferGetWidth(frame)", CVPixelBufferGetWidth(frame))
//                    print("CVPixelBufferGetHeight(frame)", CVPixelBufferGetHeight(frame))
//                    print("x", CGFloat(detectionFace[i].xMin) * 128)
//                    print("y", CGFloat(detectionFace[i].yMin) * 128)
//                    self.view1.center.x = CGFloat(detectionFace[i].xMin) * 128
//                    self.view1.center.y = CGFloat(detectionFace[i].yMin) * 128
                    self.view1.frame = CGRect(x: CGFloat(detectionFace[i].xMin) * 128, y: CGFloat(detectionFace[i].yMin) * 128, width: CGFloat(detectionFace[i].width) * 128, height: CGFloat(detectionFace[i].height) * 128)
                }
            }
            
          }
        DispatchQueue.main.async { [self] in
            self.imagePreview.image = bckgd
        }
 
       
    }
    func ResizeImage(image: UIImage, targetSize: CGSize) -> UIImage {
        let size = image.size

        let widthRatio  = targetSize.width  / image.size.width
        let heightRatio = targetSize.height / image.size.height

        // Figure out what our orientation is, and use that to form the rectangle
        var newSize: CGSize
        if(widthRatio > heightRatio) {
            newSize = CGSize(width: size.width * heightRatio, height: size.height * heightRatio)
        } else {
            newSize = CGSize(width: size.width * widthRatio,  height: size.height * widthRatio)
        }

        // This is the rect that we've calculated out and this is what is actually used below
        let rect = CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height)

        // Actually do the resizing to the rect using the ImageContext stuff
        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        image.draw(in: rect)
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        return newImage!
    }
    var documentsUrl: URL {
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    }
    private func load(){
        if let filePath = Bundle.main.path(forResource: "saveImage", ofType: "png"), let data = UIImage(contentsOfFile: filePath) {
            let bckgd = ResizeImage(image: data, targetSize: CGSize(width: 128, height: 128))
            let resultModel = modelDataHandler?.runModel(onFrame: buffer(from: bckgd)!)
              if let result = resultModel {
                  let anchorsData = faceDetection.getAnchors(options: anchors)
                  let a = Array(result)[0].value
                  let b = Array(result)[1].value
             
                  var detection: [Detection] = []
                  if a.count == 14336 {
                      detection = faceDetection.process(options: optionsFace, rawScores: b, rawBoxes: a, anchors: anchorsData)
                  } else {
                      detection = faceDetection.process(options: optionsFace, rawScores: a, rawBoxes: b, anchors: anchorsData)
                  }
                print("detection: ", detection)
                print("data: ", faceDetection.origNms(detections: detection, threshold: 0.3, img_width: 128, img_height: 128))
              }
    }
    }
//    func pixelValues(fromCGImage imageRef: CGImage?) -> (pixelValues: [Double]?, width: Int, height: Int)
//    {
//        var width = 0
//        var height = 0
//        var pixelValues: [Double]?
//        if let imageRef = imageRef {
//            width = imageRef.width
//            height = imageRef.height
//            let bitsPerComponent = imageRef.bitsPerComponent
//            let bytesPerRow = imageRef.bytesPerRow
//            let totalBytes = height * bytesPerRow
//
//            let colorSpace = CGColorSpaceCreateDeviceGray()
//            var intensities = [I](repeating: 0, count: totalBytes)
//
//            let contextRef = CGContext(data: &intensities, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: 0)
//            contextRef?.draw(imageRef, in: CGRect(x: 0.0, y: 0.0, width: CGFloat(width), height: CGFloat(height)))
//            print(intensities)
//            pixelValues =  intensities.map{data in
//            let sub = Double(data) - 127.5
//                return sub/127.5
//            }
//        }
//        return (pixelValues, width, height)
//    }

//    func image(fromPixelValues pixelValues: [Float16]?, width: Int, height: Int) -> CGImage?
//    {
//        var imageRef: CGImage?
//        if var pixelValues = pixelValues {
//            let bitsPerComponent = 8
//            let bytesPerPixel = 1
//            let bitsPerPixel = bytesPerPixel * bitsPerComponent
//            let bytesPerRow = bytesPerPixel * width
//            let totalBytes = height * bytesPerRow
//
//            imageRef = withUnsafePointer(to: &pixelValues, {
//                ptr -> CGImage? in
//                var imageRef: CGImage?
//                let colorSpaceRef = CGColorSpaceCreateDeviceGray()
//                let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue).union(CGBitmapInfo())
//                let data = UnsafeRawPointer(ptr.pointee).assumingMemoryBound(to: Float16.self)
//                let releaseData: CGDataProviderReleaseDataCallback = {
//                    (info: UnsafeMutableRawPointer?, data: UnsafeRawPointer, size: Int) -> () in
//                }
//
//                if let providerRef = CGDataProvider(dataInfo: nil, data: data, size: totalBytes, releaseData: releaseData) {
//                    imageRef = CGImage(width: width,
//                                       height: height,
//                                       bitsPerComponent: bitsPerComponent,
//                                       bitsPerPixel: bitsPerPixel,
//                                       bytesPerRow: bytesPerRow,
//                                       space: colorSpaceRef,
//                                       bitmapInfo: bitmapInfo,
//                                       provider: providerRef,
//                                       decode: nil,
//                                       shouldInterpolate: false,
//                                       intent: CGColorRenderingIntent.defaultIntent)
//                }
//
//                return imageRef
//            })
//        }
//
//        return imageRef
//    }
    
   
    func buffer(from image: UIImage) -> CVPixelBuffer? {
      let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
      var pixelBuffer : CVPixelBuffer?
      let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.size.width), Int(image.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
      guard (status == kCVReturnSuccess) else {
        return nil
      }

      CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
      let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)

      let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
      let context = CGContext(data: pixelData, width: Int(image.size.width), height: Int(image.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

      context?.translateBy(x: 0, y: image.size.height)
      context?.scaleBy(x: 1.0, y: -1.0)

      UIGraphicsPushContext(context!)
      image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
      UIGraphicsPopContext()
      CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))

      return pixelBuffer
    }
    

//    func changeToRGBA8(image: UIImage) -> UIImage? {
//      guard let cgImage = image.cgImage,
//        let data = cgImage.dataProvider?.data else { return nil }
//        let flipped = CIImage(bitmapData: data as Data,
//                              bytesPerRow: cgImage.bytesPerRow,
//                              size: CGSize(width: cgImage.width, height: cgImage.height),
//                              format: kCIFormatRGBA8,
//                              colorSpace: cgImage.colorSpace)
//        return UIImage(ciImage: flipped)
//    }
}

