// Copyright 2019 Aman Mehara
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AI_NN_ACTIVATION_H_
#define AI_NN_ACTIVATION_H_

namespace ai::nn::activation {

class activation {
  public:
    virtual double value(double input) = 0;

    virtual double derivative(double input) = 0;

    virtual ~activation() {}
};

class identity : public activation {
  public:
    double value(double input);

    double derivative(double input);
};

class relu : public activation {
  public:
    double value(double input);

    double derivative(double input);
};

class sigmoid : public activation {
  public:
    double value(double input);

    double derivative(double input);
};

class tanh : public activation {
  public:
    double value(double input);

    double derivative(double input);
};

} // namespace ai::nn::activation

#endif // AI_NN_ACTIVATION_H_
