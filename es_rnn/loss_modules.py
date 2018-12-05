import torch
import torch.nn as nn
import numpy as np

# Expression pinBallLoss(const Expression& out_ex, const Expression& actuals_ex) {//used by Dynet, learning loss function
#   vector<Expression> losses;
#   for (unsigned int indx = 0; indx<OUTPUT_SIZE; indx++) {
#     auto forec = pick(out_ex, indx);
#     auto actual = pick(actuals_ex, indx);
#     if (as_scalar(actual.value()) > as_scalar(forec.value()))
#       losses.push_back((actual - forec)*TRAINING_TAU);
#     else
#       losses.push_back((actual - forec)*(TRAINING_TAU - 1));
#   }
#   return sum(losses) / OUTPUT_SIZE * 2;
# }

#     as defined in the blog post --- https://eng.uber.com/m4-forecasting-competition/


class PinballLoss(nn.Module):

    def __init__(self, training_tau, output_size, device):
        super(PinballLoss, self).__init__()
        self.training_tau = training_tau
        self.output_size = output_size
        self.device = device

    def forward(self, predictions, actuals):
        cond = torch.zeros_like(predictions).to(self.device)
        loss = torch.sub(actuals, predictions).to(self.device)

        less_than = torch.mul(loss, torch.mul(torch.gt(loss, cond).type(torch.FloatTensor).to(self.device),
                                              self.training_tau))

        greater_than = torch.mul(loss, torch.mul(torch.lt(loss, cond).type(torch.FloatTensor).to(self.device),
                                                 (self.training_tau - 1)))

        final_loss = torch.add(less_than, greater_than)
        # losses = []
        # for i in range(self.output_size):
        #     prediction = predictions[i]
        #     actual = actuals[i]
        #     if actual > prediction:
        #         losses.append((actual - prediction) * self.training_tau)
        #     else:
        #         losses.append((actual - prediction) * (self.training_tau - 1))
        # loss = torch.Tensor(losses)
        return torch.sum(final_loss) / self.output_size * 2


# test1 = torch.rand(100)
# test2 = torch.rand(100)
# pb = PinballLoss(0.5, 100)
# pb(test1, test2)


### sMAPE

# float sMAPE(vector<float>& out_vect, vector<float>& actuals_vect) {
#   float sumf = 0;
#   for (unsigned int indx = 0; indx<OUTPUT_SIZE; indx++) {
#     auto forec = out_vect[indx];
#     auto actual = actuals_vect[indx];
#     sumf+=abs(forec-actual)/(abs(forec)+abs(actual));
#   }
#   return sumf / OUTPUT_SIZE * 200;
# }


def non_sMAPE(predictions, actuals, output_size):
    sumf = 0
    for i in range(output_size):
        prediction = predictions[i]
        actual = actuals[i]
        sumf += abs(prediction - actual) / (abs(prediction) + abs(actual))
    return sumf / output_size * 200


def sMAPE(predictions, actuals, N):
    predictions = predictions.float()
    actuals = actuals.float()
    sumf = torch.sum(torch.abs(predictions - actuals) / (torch.abs(predictions) + torch.abs(actuals)))
    return ((2 * sumf) / N) * 100


def np_sMAPE(predictions, actuals, N):
    predictions = torch.from_numpy(np.array(predictions))
    actuals = torch.from_numpy(np.array(actuals))
    return float(sMAPE(predictions, actuals, N))


### wQuantLoss

# float wQuantLoss(vector<float>& out_vect, vector<float>& actuals_vect) {
#   float sumf = 0; float suma=0;
#   for (unsigned int indx = 0; indx<OUTPUT_SIZE; indx++) {
#     auto forec = out_vect[indx];
#     auto actual = actuals_vect[indx];
#     suma+= abs(actual);
#     if (actual > forec)
#       sumf = sumf + (actual - forec)*TAU;
#     else
#       sumf = sumf + (actual - forec)*(TAU - 1);
#   }
#   return sumf / suma * 200;
# }

def wQuantLoss(predictions, actuals, output_size, training_tau):
    sumf = 0
    suma = 0
    for i in range(output_size):
        prediction = predictions[i]
        actual = actuals[i]

        suma += abs(actual)
        if (actual > prediction):
            sumf = sumf + (actual - prediction) * training_tau
        else:
            sumf = sumf + (actual - prediction) * (training_tau - 1)

    return sumf / suma * 200


# test1 = torch.rand(100)
# test2 = torch.rand(100)
# wQuantLoss(test1, test2, 100, 0.5)


### ErrorFunc

# float errorFunc(vector<float>& out_vect, vector<float>& actuals_vect) {
#   if (PERCENTILE==50)
#     return sMAPE(out_vect, actuals_vect);
#   else
#     return wQuantLoss(out_vect, actuals_vect);
# }

def errorFunc(predictions, actuals, output_size, percentile):
    if (percentile == 50):
        return sMAPE(predictions, actuals, output_size)
    else:
        return wQuantLoss(predictions, actuals, output_size, percentile / 100)


# test1 = torch.rand(100)
# test2 = torch.rand(100)
# print(errorFunc(test1, test2, 100, 48))
# print(wQuantLoss(test1, test2, 100, 0.48))
# print(errorFunc(test1, test2, 100, 50))
# print(sMAPE(test1, test2, 100))

def main():
    # Test vectorized calculation
    test1 = torch.rand(100)
    test2 = torch.rand(100)
    cpu_loss = non_sMAPE(test1, test2, 100)
    vec_loss = sMAPE(test1, test2, 100)

if __name__ == '__main__':
    main()
