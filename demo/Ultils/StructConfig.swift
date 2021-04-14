//
//  StructConfig.swift
//  movanai
//
//  Created by Nguyen Van An on 4/4/21.
//

/// A result from invoking the `Interpreter`.
struct Result {
  let inferenceTime: Double
  let inferences: [Inference]
}

/// An inference from invoking the `Interpreter`.
struct Inference {
  let confidence: Float
  let label: String
}
