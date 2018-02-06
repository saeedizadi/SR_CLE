from visdom import Visdom



class Dashboard():
    def __init__(self, port= 8097):
        self.vis = Visdom(port=port)


    def grid_plot(self, images, nrow):
        self.vis.images(images, nrow=nrow, padding=10, opts=dict(title='Results'))