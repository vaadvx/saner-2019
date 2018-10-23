It is notoriously challenging to tune the performance of deep learning neural networks.  Especially, exploratory studies have no termination condition to tell when a configuration is good enough. 

According to our experience in these experiments, a number of variations of settings
form traps for repeating the study to achieve the same good results.

* Versioning Dependencies
  * Docker is a good way to virtualise the experiment steps to make the experiments repeatable.  However, we need to control the variations:
    * Specify the version of an inherited image. Placeholders like `latest` can refer to a version different from what you had expected. So whenever possible, supply the additional tag. For examples:
```
FROM pytorch/pytorch:0.4_cuda9_cudnn7
FROM yijun/fast:v0.0.8-6
```

  * The `fast` command depends on `flatbuffers`. However, the default version of flatbuffers on Mac OSX 10.14 Homebrew is 1.7.0, while the latest version from the docker image of Alpine distribution is 1.9.0. The generated enumerators may not have exactly the same encoding of the schema we used. This difference can affect the node types in the outputted graphs.  

* Environmental Assumptions
    * The files in a same folder scanned by `srcml` may be ordered differently on different machines. This affects the ordering of AST representions of the compilation units. The problem is fixed through ordering them in our flatbuffers structure alphabetically by the file names.

    * The GGNN graphs assumes that the node types are densely distributed, otherwise the size of the matrices would increase to cover all the node types. Therefore a map is created from srcml grammar node types to a global integer, incremented when new ones are encountered FIFO. However, if the dataset files are supplied in a different order, then the mapping will change. To avoid this uncertainty, we also need to process the files alphabetically.

    * The cross-language mappings are created so that they can be aligned consistently. 
