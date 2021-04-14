//
//  ExtData.swift
//  movanai
//
//  Created by Nguyen Van An on 4/4/21.
//

import Foundation
extension Data {
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }
}
