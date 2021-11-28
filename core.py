import matplotlib.pyplot as plt
from bokeh.plotting import ColumnDataSource,figure,output_file,save
from bokeh.palettes import Spectral11
import itertools as I
plt.switch_backend('agg')
# def addvalue(dict,key,value,epoch):
#     if not key in dict.keys():
#         dict[key]=[[value]]
#     else:
#         if epoch > len(dict[key])-1:
#             dict[key].append([value])
#         else:
#             dict[key][epoch].append(value)
# def savedic(dict,fol):
#     n=1
#     numgraph=len(set([i.split(':')[0] for i in dict]))
#     axdic={}
#     fig=plt.figure()
#     for key in dict:
#         for e,i in enumerate(dict[key]):
#             if type(i)==type([]):
#                 dict[key][e]=np.mean(dict[key][e])
#     for key in dict:
#         graph,label=key.split(':')
#         if graph in axdic:
#             axdic[graph].plot(dict[key],label=f'{graph}:{label}')
#         else:
#             axdic[graph]=fig.add_subplot(numgraph,1,n)
#             n+=1
#             axdic[graph].plot(dict[key],label=f'{graph}:{label}')
#     for key in axdic:
#         axdic[key].legend()
# fig.savefig(f'{fol}/graphs.png')
# plt.close()
# with open(f'{fol}/data.pkl','wb') as f:
#     pickle.dump(dict,f)
#
# def save(model,fol,dic,argdic,title=''):
#     import json
#     savedmodelpath=f'{fol}/model.pth'
#     savedic(dic,'/'.join(savedmodelpath.split('/')[:-1]),title)
#     torch.save(model.state_dict(), savedmodelpath)
#     with open(f'{fol}/args.json','w') as f:
#         json.dump(argdic,f)
# import requests
# def send_line_notify(imagepath,notification_message='training_result'):
#     line_notify_token ='ui4IbiPkvedb3pjEGkEfSLSv1ZWyZVOxm955n41EHTy'
#     line_notify_api = 'https://notify-api.line.me/api/notify'
#     headers = {'Authorization': f'Bearer {line_notify_token}'}
#     data={'message':notification_message}
#     files = {'imageFile': open(imagepath,'rb')}
#     requests.post(line_notify_api,data=data, headers = headers, files = files)
from collections import defaultdict


class Plotter:
    def __init__(self,graphpath):
        self.graphdic = defaultdict(lambda: [[], []])
        self.graphpath=graphpath

    def addvalue(self, data, idx):
        assert type(data) == type(dict())
        for key in data:
            self.graphdic[key][0].append(idx)
            self.graphdic[key][1].append(data[key])

    def savedic(self):
        fig=plt.figure()
        graphs = defaultdict(list)
        for s in self.graphdic.keys():
            graphs[s.split(':')[0]].append(s)
        for idx,graphname in enumerate(graphs):
            ax = fig.add_subplot(len(graphs),1,idx+1)
            for key in graphs[graphname]:
                data=self.graphdic[key]
                ax.plot(data[0],data[1],label=key)
            ax.legend()
        plt.savefig(self.graphpath)
        plt.cla()
    def savebokeh(self):
        output_file(self.graphpath.replace('jpg','html'))
        TOOLTIPS=[("name","$name"),("value","$y")]
        p=figure(tooltips=TOOLTIPS)
        xs=[self.graphdic[key][0] for key in self.graphdic]
        ys=[self.graphdic[key][1] for key in self.graphdic]
        names=[key for key in self.graphdic]
        colors=getitem_num(Spectral11,len(xs))
        for idx,(x,y,name) in enumerate(zip(xs,ys,names)):
            p.line(x,y,name=name,color=colors[idx])
        save(p)

    def grad_plot(self,model,idx,tag=''):
        for name, p in model.named_parameters():
            self.addvalue({f'{tag}{name}_mean': p.grad.mean().item()},idx)
            self.addvalue({f'{tag}{name}_max': p.grad.max().item()},idx)
            self.addvalue({f'{tag}{name}_min': p.grad.min().item()},idx)

def getitem_num(L,num):
    if len(L)>num:
        return L[:num]
    else:
        return L*(num//len(L))+L[:num%len(L)]

if __name__ == '__main__':
    P = Plotter('tmp.png')
    P.addvalue({'loss:train': 0, 'loss:val': 1}, 0)
    P.addvalue({'loss:train': 1, 'acc': 2}, 1)
    P.addvalue({'loss:train': 2, 'loss:val': 4, 'acc': 0}, 2)
    P.addvalue({'loss:train': 2, 'loss:val': 4, 'acc': 4}, 3)
    P.savedic()
