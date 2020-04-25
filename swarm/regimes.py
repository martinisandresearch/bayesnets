#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"



def standard_training(num_epoch):
    optimiser = optimfunc(net.parameters(), **self.optimkwargs)
    data_out = torch.zeros(num_epoch, self.xt.shape[0])
    loss_t = torch.zeros(num_epoch)

    og_loss = self.loss_func(net(self.xt), self.yt)
    loss = 0
    for epoch in range(num_epoch):
        optimiser.zero_grad()
        ypred = net(self.xt)

        loss = self.loss_func(ypred, self.yt)

        if DEBUG:
            print(epoch, loss)

        loss_t[epoch] = loss.item()
        data_out[epoch, :] = ypred.squeeze()

        loss.backward()
        optimiser.step()
    if DEBUG:
        print(f"First loss {og_loss} v final {loss}")
    return data_out.detach(), loss_t.detach()
