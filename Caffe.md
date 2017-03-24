[TOC]





# ***Easy Caffe***

 Author:Tom Long	Email: mytomlong@gmail.com







# 1 Basic Class

## 1.1 **Blob**

### 1.1.1 Class Constructor

```cpp
Blob() : data_(), diff_(), count_(0), capacity_(0) {}	//构造函数：初始化列表 {空函数体}
explicit Blob(const int num, const int channels, const int height,const int width);		//可以通过设置数据维度（N,C,H,W）初始化
explicit Blob(const vector<int>& shape); //也可以通过传入vector<int>直接传入维数
```
### 1.1.2 Class Member

* `shared_ptr<SyncedMemory> data_`

  * > 存储前向传播的数据 

* `shared_ptr<SyncedMemory> diff_`

  * > 存储反向传递梯度

* `shared_ptr<SyncedMemory> shape_data_`

* `vector<int> shape_ `   

  * > 输入图像维度

* `int count_ `   

  * > Blob存储的元素个数（shape_所有元素乘积）

* `int capacity_` 

  * > 当前Blob的元素个数（控制动态分配）

## 1.2 **Layer**

### 1.2.1 Class Constructor

 

```cpp
//显示的构造函数不需要重写，任何初始工作在SetUp()中完成构造方法只复制层参数说明的值，如果层说明参数中提供了权值和偏置参数，也复制继承自Layer类的子类都会显示的调用Layer的构造函数

explicit Layer(const LayerParameter& param)
 : layer_param_(param), is_shared_(false) {
   // Set phase and copy blobs (if there are any).
   phase_ = param.phase();   //训练还是测试

   // 在layer类中被初始化，如果blobs_size() > 0
   // 在prototxt文件中一般没有提供blobs参数，所以这段代码一般不执行
   if (layer_param_.blobs_size() > 0) {
 	blobs_.resize(layer_param_.blobs_size());

 	for (int i = 0; i < layer_param_.blobs_size(); ++i) {
    	blobs_[i].reset(new Blob<Dtype>());
    	blobs_[i]->FromProto(layer_param_.blobs(i));
 	}
  }
}
```

### 1.2.2 Class Member

* `LayerParameter layer_param_`

  * > protobuf文件中存储的layer参数,从protocal buffers格式的网络结构说明文件中读取protected类成员，构造函数中初始化


* `Phase phase_`

  * >  The phase: TRAIN or TEST  层状态，参与网络的训练还是测试


* `vector<shared_ptr<Blob<Dtype> > > blobs_`

  * > The vector that stores the learnable parameters as a set of blobs. 可学习参数层权值和偏置参数，使用向量是因为权值参数和偏置是分开保存在两个blob中的.在基类layer中初始化(只是在描述文件定义了的情况下)


* `vector<bool> param_propagate_down_`

  * > 标志每个可学习参数blob是否需要计算反向传递的梯度值


* `vector<Dtype> loss_`

  * > The vector that indicates whether each top blob has a non-zero weight in the objective function.非LossLayer为零，LossLayer中表示每个top blob计算的loss的权重

* `bool is_shared_`

  * > Whether this layer is actually shared by other nets

* `shared_ptr<boost::mutex> forward_mutex_`

  * > The mutex for sequential forward if this layer is shared.类型为 boost::mutex 的 mutex 全局互斥对象若该layer被shared，则需要这个mutex序列保持forward过程的正常运行

## 1.3 **Net**

### 1.3.1 Class Constructor

```cpp
  explicit Net(const NetParameter& param, const Net* root_net = NULL);
  explicit Net(const string& param_file, Phase phase,const Net* root_net = NULL);
```

### 1.3.2 Class Member

* `string name_`

  * > 网络名称


* `Phase phase_`

  * > 测试还是训练


* `vector<shared_ptr<Layer<Dtype> > > layers_`

  * > Layer容器


* `vector<string> layer_names_`

  * > 每层layer的名称


* `map<string, int> layer_names_index_`

  * > 关联容器，layer名称所对应的索引


* `vector<bool> layer_need_backward_`

  * > 每层layer是否需要计算反向传导


* `vector<shared_ptr<Blob<Dtype> > > blobs_`

  * >  blobs_存储的是中间结果，是针对整个网络中所有非参数blob而设计的一个变量


* `vector<string> blob_names_`

  * >  整个网络中，所有非参数blob的name


* `map<string, int> blob_names_index_`

  * >  blob 名称索引键值对


* `vector<bool> blob_need_backward_`

  * >   整个网络中，所有非参数blob，是否需要backward。注意，这里所说的所有非参数blob其实指的是AppendTop函数中遍历的所有top blob,并不是每一层的top+bottom,因为这一层的top就是下一层的bottom,网络是一层一层堆起来的。


* `vector<vector<Blob<Dtype>*> > bottom_vecs_`

  * >   存储整个网络所有网络层的bottom blob指针,实际上存储的是前一层的top，因为网络是一层一层堆起来的


* `vector<vector<int> > bottom_id_vecs_`

  * >  存储整个网络所有网络层的bottom blob的ID


* `vector<vector<bool> > bottom_need_backward_`

  * >  整个网络所有网络层的bottom blob是否需要backward


* `vector<vector<Blob<Dtype>*> > top_vecs_`

  * > 存储整个网络所有网络层的top blob指针


* `vector<vector<int> > top_id_vecs_`

  * > 存储整个网络所有网络层的top blob的ID.top_id_vecs_中存储的最基本元素是 blob_id：每一个新的blob都会赋予其一个blob_id,top_vecs_则与之对应，但是这个blob_id可能是会有重复的（因为in-place）


* `vector<Dtype> blob_loss_weights_`

  * >  每次遍历一个layer的时候，都会resize blob_loss_weights_, 然后调用模板类layer的loss函数返回loss_weight


* `vector<vector<int> > param_id_vecs_`

  * > 存储每层的可学习参数id存储的基本元素是net_param_id，每遍历一个参数blobnet_param_id和param_id_vecs_都会更新


* `vector<int> param_owners_`

  * >  表示参数所属的layer在layers_中的位置param_owners_ 是一个存储parameter "onwer"的一个向量  ——> -1表示当前Layer就是该parameter的"owner"

* `vector<string> param_display_names_`

* `param_layer_indices_vector<pair<int, int> > param_layer_indices_`

  * > 其元素为当layer_id 与当前param_id 组成的pair.vector<pair<int, int> > param_layer_indices_


* `map<string, int> param_names_index_`

  * > 是整个网络的参数non-empty name与index的映射。注意，这个name是ParamSpec 类型中的name


* `vector<int> net_input_blob_indices_`

*  `vector<int> net_output_blob_indices_`

  * > 整个网络的输入输出blob的ID


* `vector<Blob<Dtype>*> net_input_blobs_`

* `vector<Blob<Dtype>*> net_output_blobs_`

  * >  网络输入输出的所有blob


*  `vector<shared_ptr<Blob<Dtype> > > params_`

  * >  网络中的所有参数
    >  整个网络的参数blob。 !!!不管这个参数有没有non-empty name，是否参与share!!!


* `vector<Blob<Dtype>*> learnable_params_`


* `vector<int> learnable_param_ids_`

  * > The mapping from params_ -> learnable_params_: we have learnable_param_ids_.size() == params_.size(),and learnable_params_[learnable_param_ids_[i]] == params_[i].get() if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer and learnable_params_[learnable_param_ids_[i]] gives its owner.

/// the learning rate multipliers for learnable_params_

* `vector<float> params_lr_`
* ` vector<bool> has_params_lr_`


* `vector<float> params_weight_decay_`

  * > the weight decay multipliers for learnable_params_

* `vector<bool> has_params_decay_`


* `size_t memory_used_`

  * >   存储网络所用的字节数


* ` bool debug_info_` 

  * >  Whether to compute and display debug info for the net.


* `const Net* const root_net_`

  * >  The root net that actually holds the shared layers in data parallelism

## 1.4 **Solver**

### 1.4.1 Class Constructor

```cpp
 explicit Solver(const SolverParameter& param,const Solver* root_solver = NULL);
 explicit Solver(const string& param_file, const Solver* root_solver = NULL);
```
### 1.4.2 Class Member

* `SolverParameter param_`

  * > Solver参数

* `int iter_`

* `int current_step_`


* `shared_ptr<Net<Dtype> > net_`

  * > 训练网络，有且只有一个



* `vector<shared_ptr<Net<Dtype> > > test_nets_`

  * > 测试网络可以有多个

* `vector<Callback*> callbacks_`

* `vector<Dtype> losses_`

* `Dtype smoothed_loss_`


* `const Solver* const root_solver_`

  * > 在数据并行中，继续根solver层保持根nets（包含共享的层）


* `ActionCallback action_request_function_`

  * >  通过函数是选择确认按钮来选择保存还是退出快照。


* ` bool requested_early_exit_`

  * >   True iff a request to stop early was received.



















