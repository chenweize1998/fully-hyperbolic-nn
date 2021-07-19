class P_sum(Function):
    @staticmethod
    def forward(ctx, x, y):
        x2 = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
        y2 = torch.clamp(torch.sum(y * y, dim=-1, keepdim=True), 0, 1-1e-5)

        xy = torch.sum(x * y, dim=-1, keepdim=True)
        one_p_2xy = 2 * xy + 1
        denominator = (one_p_2xy + x2*y2)
        rate_x = (one_p_2xy + y2) / denominator
        rate_y = (x2 - 1) / denominator
        numerator = rate_x * x - rate_y * y
        ctx.save_for_backward(x.detach(), y.detach(), x2.detach(), y2.detach(), one_p_2xy.detach(), (rate_x/denominator).detach(), (rate_y/denominator).detach())
        return numerator

    @staticmethod
    def backward(ctx, grad):
        x, y, x2, y2, one_p_2xy, rate_x, rate_y, = ctx.saved_tensors

        x_right = torch.sum((x * grad), dim = -1, keepdim = True) * 2
        y_right = torch.sum((y * grad), dim = -1, keepdim = True) * 2

        yx_right = rate_y * x_right
        yy_right = rate_y * y_right
        xx_right = rate_x * x_right
        xy_right = rate_x * y_right

        a = (yx_right * y2 + yy_right)
        b = (xx_right * y2 + xy_right)
        c = (one_p_2xy * yx_right - x2 * yy_right)
        
        grad_x = (a * y \
                 - b * x \
                 + grad * rate_x) #* denominator

        grad_y = (a * x \
                 - c * y \
                 - grad * rate_y) #* denominator

        return grad_x, grad_y

def cross_p_dis(x, y):

    x2 = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    y2 = torch.clamp(torch.sum(y * y, dim=-1, keepdim=True), 0, 1-1e-5).transpose(1,0)
    dotxy = torch.mm(x, y.transpose(1,0))
    rate_x = (1+2*dotxy+y2)
    rate_y = (1 - x2)
    numerator = rate_x * rate_x * x2 + rate_y * rate_y * y2 + 2 * rate_x * rate_y * dotxy
    denominator = 1 + 2 * dotxy + x2 * y2
    
    return (numerator ** 0.5) / denominator