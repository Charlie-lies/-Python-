import tkinter
from tkinter import *
from tkinter import ttk, Frame, Tk, messagebox, Menu
from PIL import Image, ImageDraw
import pickle
import csv
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return x*(1-x)

class NeuralNetwork:
    def __init__(self,layers):  # (64,100,50,10)
        # 权值的初始化，范围-1到1
        self.U = np.random.random((layers[0]+1,layers[1]+1))*2-1
        self.V = np.random.random((layers[1]+1,layers[2]+1))*2-1
        self.W = np.random.random((layers[2]+1,layers[3]))*2-1
        
    def train(self,X,y,X_test,y_test,lr=0.11,epochs=10000):
        # 添加偏置
        temp = np.ones([X.shape[0],X.shape[1]+1])
        temp[:,0:-1] = X  # 最后一列都是1
        X = temp
        
        for n in range(epochs+1):
            i = np.random.randint(X.shape[0]) # 随机选取一个数据
            x = [X[i]]
            x = np.atleast_2d(x)  # 转为2维数据
            
            L0 = sigmoid(np.dot(x,self.U))
            L1 = sigmoid(np.dot(L0,self.V))  # 隐层输出
            L2 = sigmoid(np.dot(L1,self.W))  # 输出层输出
            
            L2_delta = (y[i]-L2)*dsigmoid(L2)
            L1_delta= L2_delta.dot(self.W.T)*dsigmoid(L1)
            L0_delta= L1_delta.dot(self.V.T)*dsigmoid(L0)

            self.W += lr*L1.T.dot(L2_delta)
            self.V += lr*L0.T.dot(L1_delta)
            self.U += lr*x.T.dot(L0_delta)
            
            #每训练1000次预测一次准确率
            if n%1000==0:
                predictions = []
                for j in range(X_test.shape[0]):
                    o = self.predict(X_test[j])
                    predictions.append(np.argmax(o))#获取预测结果
                self.accuracy = np.mean(np.equal(predictions,y_test))
                print('epoch:',n,'accuracy:',self.accuracy)
        
    def predict(self,x):
        #添加偏置
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        x = temp
        x = np.atleast_2d(x)#转为2维数据
        
        L0 = sigmoid(np.dot(x,self.U))
        L1 = sigmoid(np.dot(L0,self.V))#隐层输出
        L2 = sigmoid(np.dot(L1,self.W))#输出层输出
        
        return L2



class Window(Frame):
    
    def __init__(self, master= None):
        super().__init__()
        self.master = master
        self.init_window()
        
        # 记录最后绘制图形的id
        self.lastDraw = 0
        
        # 前景色
        self.foreColor = '#000000'
        self.backColor = '#FFFFFF'
        
        #控制是否允许画图的变量，1：允许，0：不允许
        self.yesno = tkinter.IntVar(value=0)
        #控制画图类型的变量 
        self.what = tkinter.IntVar(value=1)

        # 记录鼠标位置的变量
        self.X = tkinter.IntVar(value=0)
        self.Y = tkinter.IntVar(value=0)

        self.samples = np.array([])  # 保存手写数字的样本
        self.labels = np.array([])  # 保存样本对应的数字标签

        
    def init_window(self):
        self.master.title('手写数字识别demo-by 查尔Char')
        
        menubar = Menu(self.master)
        self.master.config(menu=menubar)
        
        menu = Menu(menubar)
        menu.add_command(label="训练新的模型", command=self.retrainning)
        menu.add_command(label="训练新的模型（含新样本）", command=self.train_with_newsample)
        menu.add_command(label="关于", command=self.aboutme)
        menubar.add_cascade(label="菜单", menu=menu)

        self.frame_info = ttk.LabelFrame(self.master, text='Info: ' )
        self.frame_info.place(x=15,y=0)
        self.infoLabel = ttk.Label(self.frame_info, text="使用提示: 写字→载入模型→数字识别→保存样本(可选)",anchor="center",font=("微软雅黑",9))
        self.infoLabel.pack(fill=tkinter.BOTH, expand=tkinter.YES)
        
        self.frame_pad = ttk.LabelFrame(self.master, text="写字区")
        self.frame_pad.place(x=10, y=50, width=200, height=200)
        
        # 创建画布
        image = tkinter.PhotoImage()
        self.canvas = tkinter.Canvas(self.frame_pad, bg='white', width=200, height=200)
        self.canvas.create_image(120, 120, image=image)
        self.canvas.bind('<B1-Motion>', self.onLeftButtonMove)
        self.canvas.bind('<Button-1>', self.onLeftButtonDown)
        self.canvas.bind('<ButtonRelease-1>', self.onLeftButtonUp)
        self.canvas.pack(fill=tkinter.BOTH, expand=tkinter.YES)
        self.base = Image.new("RGB", (200, 200), (255,255,255))
        self.d = ImageDraw.Draw(self.base)
        
        action_frame = ttk.Frame(root)
        action_frame.place(x=225,y=65,width=70,height=150)
        button_cl = ttk.Button(action_frame, text="重写", command=self.Clear)        
        button_cl.pack(pady=5)
        button_start = ttk.Button(action_frame, text="载入模型", command=self.load_model)
        button_start.pack(pady=15)
        button_reg = ttk.Button(action_frame, text="数字识别", command=self.predict)
        button_reg.pack(pady=5)  
        
        
        self.frame2 = ttk.LabelFrame(self.master, text="数字识别结果")
        self.frame2.place(x=320, y=50, width=150, height=150)
        image2 = tkinter.PhotoImage()
        self.canvas2 = tkinter.Canvas(self.frame2, bg='white', width=200, height=200)
        self.canvas2.create_image(120, 120, image=image2)
        self.canvas2.pack(fill=tkinter.BOTH, expand=tkinter.YES)
        
        self.label = ttk.Label(self.master, text="输入数字作\n为样本标签",anchor="center",font=("微软雅黑",10))
        self.label.place(x=290,y=215,width=120,height=60)
        self.numEntry = Entry(self.master)
        self.numEntry.place(x=250,y=230,width=50,height=30)
        button_save = ttk.Button(self.master, text="保存样本", command=self.saveSample)
        button_save.place(x=400,y=230,width=70,height=30)
        


    # 按住鼠标左键移动，画图
    def onLeftButtonMove(self,event):
        # global lastDraw
        if self.yesno.get()==0:
            return
        if self.what.get()==1:
            #使用当前选择的前景色绘制曲线
            # canvas.create_line(X.get(), Y.get(), event.x, event.y, width=8, fill=foreColor)
            self.canvas.create_oval(self.X.get(), self.Y.get(), event.x, event.y, width=8, fill=self.foreColor)
            self.d.line([self.X.get(), self.Y.get(), event.x, event.y],
                    width=8,
                    fill='black')

            self.X.set(event.x)
            self.Y.set(event.y)
        
        # 鼠标左键单击，允许画图
    def onLeftButtonDown(self,event):
        self.yesno.set(1)
        self.X.set(event.x)
        self.Y.set(event.y)
        if self.what.get()==4:
            self.canvas.create_text(event.x, event.y, text=text)

    # 鼠标左键抬起，不允许画图
    def onLeftButtonUp(self,event):
        self.yesno.set(0)
        self.lastDraw = 0
        
    # 添清除
    def Clear(self):
        # pillow的img对象重新画成白色的
        self.d.rectangle([0,0,200,200],fill='white')
        
        # 删除tkinter canvas的所有对象
        for item in self.canvas.find_all():
            self.canvas.delete(item)

        for item in self.canvas2.find_all():
            self.canvas2.delete(item)

        self.label['text'] = "输入数字作\n为样本标签"
        self.label['foreground'] = ['black']

    def trainning(self,newSample=None):
        digits = load_digits()#载入数据
        X = digits.data#数据
        y = digits.target#标签
        #输入数据归一化
        X -= X.min()
        X /= X.max()

        if newSample is not None:
            X = np.concatenate((X, newSample[:,:-1]))
            y = np.concatenate((y, newSample[:,-1].astype(int)))
            
        nm = NeuralNetwork([64,100,50,10])#创建网络

        X_train,X_test,y_train,y_test = train_test_split(X,y)
        # labels_train = LabelBinarizer().fit_transform(y_train)
        # labels_test = LabelBinarizer().fit_transform(y_test)
        labels_train = np.eye(10)[y_train].astype(np.int16)
        labels_test = np.eye(10)[y_test].astype(np.int16)

        print('start')
        nm.train(X_train,labels_train,X_test,y_test,epochs=20000)
        print('end')
        
        return nm


    def train_with_newsample(self):
        try:
            new_samples = np.genfromtxt('mysamples.csv', delimiter=',')
            
            if len(new_samples) > 5:
                self.model = self.trainning(newSample=new_samples)
            else:
                self.infoLabel['text'] = '提示：mysamples.csv文件没有足够的新样本'
                self.infoLabel['foreground'] = ['red']
        except:
            self.infoLabel['text'] = '提示：当前目录未找到mysamples.csv文件，或数据格式有误！'
            self.infoLabel['foreground'] = ['red']
        
        
    def retrainning(self):
        print('retrainning...')
        self.infoLabel['text'] = '提示：正在训练新的模型。。。'
        self.infoLabel['foreground'] = ['blue']
        self.model = self.trainning()
            
        with open('nmModel.pkl', 'wb') as pkl:
            pickle.dump(self.model, pkl, pickle.HIGHEST_PROTOCOL)
            self.infoLabel['text'] = '提示：新模型训练完成！'
            self.infoLabel['foreground'] = ['blue']

        
    def load_model(self):
        try:
            pkl = open('nmModel.pkl', 'rb')
            self.model = pickle.load(pkl)
            
        except:
            self.infoLabel['text'] = '提示：未找到本地模型，正在训练新的模型。'
            self.infoLabel['foreground'] = ['red']
            self.model = self.trainning()
            
            with open('nmModel.pkl', 'wb') as pkl:
                pickle.dump(self.model, pkl, pickle.HIGHEST_PROTOCOL)
        finally:
            self.infoLabel['text'] = '提示：模型加载完成'
            self.infoLabel['foreground'] = ['blue']
            

    def predict(self):
        
        preproces = self.pre_job()
        if not preproces:
            return
        
        for item in self.canvas2.find_all():
            self.canvas2.delete(item)

        try:
            result = self.model.predict(self.test)
        except AttributeError:
            self.canvas2.create_text(18, 65,
            text = '模型未加载\n或加载失败\n请重载模型',
            font = ("微软雅黑", 16, "bold"),
            fill= "red",
            anchor = W,
            justify = LEFT)

            return
        
        print(np.argmax(result))
        
        titleFont = ("微软雅黑", 50, "bold")
        self.canvas2.create_text(45, 65,
            text = np.argmax(result),
            font = titleFont,
            fill= "Turquoise",
            anchor = W,
            justify = LEFT)

    def pre_job(self):
        img = self.base
        x,y = img.size
        img = img.convert('L')
        raw_data = img.load()
        
        "这里有点奇怪，横纵颠倒了？"
        "行列和横纵坐标"
        L = [[raw_data[j, i] for j in range(img.size[0])] for i in range(img.size[1])]
        L_arry = np.array(L)
        print(L_arry.shape)
        
        row_member = L_arry.sum(axis=1) < 245*img.size[0]
        col_member = L_arry.sum(axis=0) < 245*img.size[1]

        # 图片裁剪的边缘
        r_cs = row_member.cumsum()
        
        if r_cs.max() < 2:
            # 过滤少于2行非白色像素，即没有画数字的情况
            self.canvas2.create_text(15, 38,
            text = '请先在写字\n区写数字',
            font = ("微软雅黑", 18, "bold"),
            fill= "red",
            anchor = W,
            justify = LEFT)
            
            return
            
        y_min =np.argwhere(r_cs == 1)[0,0] - 1 # 第一个非纯白的列
        y_max = r_cs.argmax() + 1

        c_cs = col_member.cumsum()
        x_min = np.argwhere(c_cs == 1)[0,0] - 1
        x_max = c_cs.argmax() + 1
        
        # 要裁剪成矩形，需要检查一下横竖边
        x_len = x_max - x_min
        y_len = y_max - y_min
        if y_len - x_len > 0:
            x_min = x_min - int(1/2 * (y_len - x_len))
            if x_min < 0:
                x_min = 0
            x_max = x_min + y_len
        elif y_len - x_len < 0:
            y_min = y_min - int(1/2 * (x_len - y_len))
            if y_min < 0:
                y_min = 0
            y_max = y_min + x_len
            
        new = img.crop((x_min, y_min, x_max, y_max))
        print(new.size)
        
        new = new.resize((8, 8))  # 裁剪成和训练数据一样的尺寸
        new_data = new.load()
        new_array = np.array([[new_data[j, i] for j in range(8)] for i in range(8)])
        print(new_array.shape)
        
        test = (255 - new_array) / 255
        
        self.test = np.r_[test.ravel()]
            
        print(self.test)

        return 'Done'


    def saveSample(self):
        target = self.numEntry.get()
        print(type(target))
        
        try:
            print(self.test.shape)
        except:
            preproces = self.pre_job()
            if not preproces:
                return
        
        if target.isdigit():
            print(self.test.shape)
            if self.test.shape != (64,):
                self.label['text'] = '样本数据格式\n有误，请重试'
                self.label['foreground'] = ['red']
                return
            
            print(target)
            self.label['foreground'] = ['black']
            
            with open('mysamples.csv','a', newline='') as csvfile:
                # obj = self.test.append(int(target))
                obj = self.test.tolist()
                obj.append(int(target))
                print(obj)
                writer = csv.writer(csvfile)
                writer.writerow(obj)
                print('Save!')
                
                self.label['text'] = '保存成功'
                self.label['foreground'] = ['blue']

        else:
            self.label['text'] = '先输入数字作\n为样本的标签'
            self.label['foreground'] = ['red']

            
    def aboutme(self):
        messagebox.showinfo("关于","\n\n基于神经网络和sklearn的digit数据集编写的手写数字识别demo\n\n"
                                   "Recognizer for handwritten numeral-v0.01                      \n"
                                   "                                                  —— by 查尔Char\n\n"
                                  )         

    
if __name__ == '__main__':
    
    root = Tk()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    ww = 500
    wh = 300
    x = (sw-ww) / 2 - 100
    y = 200
    root.geometry("%dx%d+%d+%d" % (ww, wh, x, y))
    app = Window(root)
    root.mainloop()
