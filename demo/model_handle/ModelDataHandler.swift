import CoreImage
import TensorFlowLite
import UIKit
import Accelerate


class ModelDataHandler {
    let threadCount: Int
    let threadCountLimit = 10
    
    // MARK: - Model Parameters
    var batchSize = 1
    var inputChannels: Int = 3
    var inputWidth = 128
    var inputHeight = 128
    // MARK: - Private Properties
    private var interpreter: Interpreter
    private let alphaComponent = (baseOffset: 4, moduloRemainder: 3)
    private var modelFileInfo: FileInfo?
    
    init?(modelFileInfo: FileInfo, threadCount: Int = 1) {
        let modelFilename = modelFileInfo.name
        self.modelFileInfo = modelFileInfo
        guard let modelPath = Bundle.main.path(
            forResource: modelFilename,
            ofType: modelFileInfo.extension
        ) else {
            print("Khong the load modal: \(modelFilename).")
            return nil
        }
        
      
        
        // Specify the options for the `Interpreter`.
        self.threadCount = threadCount
        var options = Interpreter.Options()
        options.threadCount = threadCount
        do {
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            try interpreter.allocateTensors()
        } catch let error {
            print("khong the tao interpreter vi: \(error.localizedDescription)")
            return nil
        }
    }
    
    
    func convert(cmage:CIImage) -> UIImage
    {
        let context:CIContext = CIContext.init(options: nil)
        let cgImage:CGImage = context.createCGImage(cmage, from: cmage.extent)!
        let image:UIImage = UIImage.init(cgImage: cgImage)
        return image
    }
    
 
    /// Performs image preprocessing, invokes the `Interpreter`, and processes the inference results.
    func runModel(onFrame pixelBuffer: CVPixelBuffer) -> [String:[Double]] {
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
                sourcePixelFormat == kCVPixelFormatType_32BGRA ||
                sourcePixelFormat == kCVPixelFormatType_32RGBA)
        
//        set size day vao modal
        let scaledSize = CGSize(width: inputWidth, height: inputHeight)
        guard let thumbnailPixelBuffer = pixelBuffer.centerThumbnail(ofSize: scaledSize) else {
            return [:]
        }
//
        do {
            
            let inputTensor = try interpreter.input(at: 0)

            // Remove the alpha component from the image buffer to get the RGB data.
            guard let rgbData = rgbDataFromBuffer(
                thumbnailPixelBuffer,
                byteCount: batchSize * CVPixelBufferGetWidth(thumbnailPixelBuffer) * CVPixelBufferGetHeight(pixelBuffer) * inputChannels,
                isModelQuantized: inputTensor.dataType == .uInt8
            ) else {
                print("Khong the conver image to rgb data")
                return [:]
            }
            
            // Copy the RGB data to the input `Tensor`.
            try interpreter.copy(rgbData, toInputAt: 0)
            
            // Run inference by invoking the `Interpreter`.
            let startDate = Date()
            try interpreter.invoke()
            let interval = Date().timeIntervalSince(startDate) * 1000
            
            var results: [String:[Double]] = [:]
            for i in 0..<interpreter.outputTensorCount {
               let outputTensor = try interpreter.output(at: i)
                var tempOutPut: [Double] = []
                switch outputTensor.dataType {
                case .uInt8:
                    guard let quantization = outputTensor.quantizationParameters else {
                        return [:]
                    }
                    let quantizedResults = [UInt8](outputTensor.data)
                    tempOutPut = quantizedResults.map {
                        Double(quantization.scale) * Double(Int($0) - quantization.zeroPoint)
                    }
                case .float32:
                    let quantizedResults = [Float32](unsafeData: outputTensor.data) ?? []
                    tempOutPut = quantizedResults.map{data in return Double(data)}
                default:
                    print("kieu data \(outputTensor.dataType) is khong ho tro")
                    tempOutPut = []
                }
                results[outputTensor.name] = tempOutPut
            }
            return results
        } catch let error {
            print("khong the cap phat interpreter voi loi : \(error.localizedDescription)")
            return [:]
        }
    }
    
    private func rgbDataFromBuffer(
        _ buffer: CVPixelBuffer,
        byteCount: Int,
        isModelQuantized: Bool
    ) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        }
        guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let destinationChannelCount = 3
        let destinationBytesPerRow = destinationChannelCount * width
        
        var sourceBuffer = vImage_Buffer(data: sourceData,
                                         height: vImagePixelCount(height),
                                         width: vImagePixelCount(width),
                                         rowBytes: sourceBytesPerRow)
        
        guard let destinationData = malloc(height * destinationBytesPerRow) else {
            print("Error: tran bo dem")
            return nil
        }
        
        defer {
            free(destinationData)
        }
        
        var destinationBuffer = vImage_Buffer(data: destinationData,
                                              height: vImagePixelCount(height),
                                              width: vImagePixelCount(width),
                                              rowBytes: destinationBytesPerRow)
        
        let pixelBufferFormat = CVPixelBufferGetPixelFormatType(buffer)
        
        switch (pixelBufferFormat) {
        case kCVPixelFormatType_32BGRA:
            vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        case kCVPixelFormatType_32ARGB:
            vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        case kCVPixelFormatType_32RGBA:
            vImageConvert_RGBA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        default:
            // Unknown pixel format.
            return nil
        }
        
        let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
        if isModelQuantized {
            return byteData
        }
        
        // Not quantized, convert to floats
        let bytes = Array<UInt8>(unsafeData: byteData)!
        var floats = [Float]()
        for i in 0..<bytes.count {
            floats.append((Float(bytes[i]) - 127.5) / 127.5)
           
        }
//        print(floats)
        return Data(copyingBufferOf: floats)
    }
}
