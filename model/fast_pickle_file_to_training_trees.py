"""Parse trees from a data source."""
import ast
import sys

import cPickle as pickle
import random
from collections import defaultdict

"""This function is to convert the original tree in a pickle to a list of node with children
tree => {node: kind, chilren: []}
For example, if a tree has 10 nodes, then this function will convert the tree into an array of 10 json object, each of object contains a node with the list of its children
In this case, we can treat the children as the context of a node"""
def parse_pickle_to_training_trees(infile,outfile):
    """Parse trees with the given arguments."""
    print ('Loading pickle file')

    sys.setrecursionlimit(1000000)
    with open(infile, 'rb') as file_handler:
        data_source = pickle.load(file_handler)
    
    print('Pickle file load finished')

    train_samples = []
    test_samples = []

    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    for item in data_source:

        tree = item['tree']
       
        label = item['metadata']['label']
      
        if tree.HasField("element"):
            root = tree.element
            sample, size = _traverse_tree(root)
        
        if size > 10000 or size < 200:
            continue

        roll = random.randint(0, 100)

        datum = {'tree': sample, 'label': label}

        if roll < 20:
            test_samples.append(datum)
            test_counts[label] += 1
        else:
            train_samples.append(datum)
            train_counts[label] += 1

    random.shuffle(train_samples)
    random.shuffle(test_samples)

    # create a list of unique labels in the data
    labels = list(set(train_counts.keys() + test_counts.keys()))

    print('Dumping sample')
    with open(outfile, 'wb') as file_handler:
        pickle.dump((train_samples, test_samples, labels), file_handler)
        file_handler.close()
    print('dump finished')
    print('Sampled tree counts: ')
    print('Training:', train_counts)
    print('Testing:', test_counts)


"""This function is to convert the original tree in a pickle to a list of node with context
tree => {node: kind, : context: []}
The context is extracted from the slicing part from f-ast tools. More specifically, the def-use information will be used"""

def parse_pickle_to_training_trees_with_dependency(infile,outfile_def_use):
    """Parse trees with the given arguments."""
    # TODO here
    print ('Loading pickle file and generating tree with dependency')

    sys.setrecursionlimit(1000000)
    with open(infile, 'rb') as file_handler:
        data_source = pickle.load(file_handler)
    # print data_source
    print('Pickle file load finished')


    train_samples_def_use = []
    test_samples_def_use = []

    train_counts_def_use = defaultdict(int)
    test_counts_def_use = defaultdict(int)

 
    size_array = []
    size_def_use_array = []

    count = 0
    for item in data_source:
        print "-------------"
        tree = item['tree']
        # print tree
        label = item['metadata']['label']
        
        if tree.HasField("element"):
            root = tree.element

            line_dict, use_dict, def_dict = _traverse_tree_to_extract_dependencies(root)
            tree_nodes_def_use, size_def_use = _traverse_tree_to_add_def_use(root,line_dict,use_dict,def_dict)
          
         
            size_def_use_array.append(size_def_use)
        
            count+=1
       
        if size_def_use_array > 12000 or size < 150:
        # if size_def_use > 10000
            continue

        roll = random.randint(0, 100)

        datum = {'tree': tree_nodes, 'label': label}

        datum_def_use = {'tree': tree_nodes_def_use, 'label': label}

        if roll < 20:
            test_samples.append(datum)
            test_counts[label] += 1

            test_samples_def_use.append(datum_def_use)
            test_counts_def_use[label] += 1

        else:
            train_samples.append(datum)
            train_counts[label] += 1

            train_samples_def_use.append(datum_def_use)
            train_counts_def_use[label] += 1

    # random.shuffle(train_samples)
    # random.shuffle(test_samples)
   
    # create a list of unique labels in the data
    labels = list(set(train_counts.keys() + test_counts.keys()))

    print('Dumping trees....')
  
    with open(outfile_def_use, 'wb') as file_handler:
       pickle.dump((train_samples_def_use, test_samples_def_use, labels), file_handler,protocol=2)
       file_handler.close()


    print("Max tree size : " + str(max(size_array)))
    print("Max tree def_use size : " + str(max(size_def_use_array)))
    print('dump finished')
    print('Sampled tree counts: ')
    print('Training:', train_counts)
    print('Testing:', test_counts)
    return 


def _traverse_tree_to_extract_dependencies(root):
    num_nodes = 1
    queue = [root]

    line_dict = {}
    def_dict = {}
    use_dict = {}

    root_json = {
        "node": str(root.kind),

        "children": []
    }
    queue_json = [root_json]
    use_count = 0
    def_count = 0 
    while queue:
      
        current_node = queue.pop(0)
        num_nodes += 1
        # print (_name(current_node))
        current_node_json = queue_json.pop(0)


        children = [x for x in current_node.child]
        queue.extend(children)
        
        if(if_node_contains_def(current_node)):
            def_count +=1
            # print current_node
            def_node = extract_def(current_node)
            def_key = "def_" + str(def_node.text)
          
            if def_key in def_dict:

                value = def_dict[def_key]
                value.append(current_node)
                def_dict[def_key] = value
            else:
                arr = []
                arr.append(current_node)
                def_dict[def_key] = arr


        if(if_node_contains_use(current_node)):
            use_count +=1
            # print current_node
            use_node = extract_use(current_node)
            use_key = "use_" + str(use_node.text)
          
            if use_key in use_dict:

                value = use_dict[use_key]
                value.append(current_node)
                use_dict[use_key] = value
            else:
                arr = []
                arr.append(current_node)
                use_dict[use_key] = arr

                
        line_key = "line_" + str(current_node.line)
        if line_key in line_dict:

            value = line_dict[line_key]
            value.append(current_node_json)
            line_dict[line_key] = value
        else:
            arr = []
            arr.append(current_node_json)
            line_dict[line_key] = arr

       
        for child in children:
            # print "##################"
            #print child.kind

            child_json = {
                "node": str(child.kind),
                "children": []
            }
            # print "--------------------"
            # print child.kind
            current_node_json['children'].append(child_json)
            queue_json.append(child_json)


    # print len(use_dict["use_1"])
    # print use_dict["use_1"]

    return line_dict, use_dict, def_dict

def extract_def(node):
    def_node = None
    children = [x for x in node.child]   
    for child in children:
        if child.kind == 386:
            def_node = child
    return def_node
    


def if_node_contains_def(node):
    check = False
    children = [x for x in node.child]   
    for child in children:
        if child.kind == 386:
            check = True
    return check


def extract_use(node):
    def_node = None
    children = [x for x in node.child]   
    for child in children:
        if child.kind == 387:
            def_node = child
    return def_node
    


def if_node_contains_use(node):
    check = False
    children = [x for x in node.child]   
    for child in children:
        if child.kind == 387:
            check = True
    return check



def _traverse_tree_to_add_def_use(root,line_dict,use_dict, def_dict):
    num_nodes = 1
    queue = [root]

    root_json = {
        "node": str(root.kind),

        "children": []
    }
    queue_json = [root_json]
    while queue:
       
        current_node = queue.pop(0)
        num_nodes += 1
        # print (_name(current_node))
        current_node_json = queue_json.pop(0)


        children = [x for x in current_node.child]
        queue.extend(children)
        


        for child in children:
       
         

            child_json = {
                "node": str(child.kind),
                "children": []
            }
           
            if child.kind == 386:
                # Which means the current node is a slice define node
                slicing_part = child.text
                # print "slicing part : " + slicing_part
                # print get_all_keys(use_dict)
                use_key = "use_" + slicing_part
                if use_key in use_dict:
                    use_arr = use_dict[use_key]

                    for use in use_arr:
                        # print "use at line : " + str(use.line)
                        use_line = use.line
                        use_line_nodes = line_dict["line_" + str(use_line)]
                        # print use_line_nodes
                        # print "use line nodes : ---------------- " + str(use_line_nodes)
                        current_node_json['children'].extend(use_line_nodes)
                        num_nodes += len(use_line_nodes)
                        # print "current : ------------------------" + str(current_node_json["children"])
                else:
                    current_node_json['children'].append(child_json)
                    # print "current : ------------------------" + str(current_node_json["children"])
            if child.kind == 387:
                slicing_part = child.text
                # print "slicing part : " + slicing_part
                # print get_all_keys(use_dict)
                def_key = "def_" + slicing_part
                if def_key in def_dict:
                    def_arr = def_dict[def_key]

                    for de in def_arr:
                        # print "use at line : " + str(use.line)
                        def_line = de.line
                        def_line_nodes = line_dict["line_" + str(def_line)]
                        # print use_line_nodes
                        # print "use line nodes : ---------------- " + str(use_line_nodes)
                        current_node_json['children'].extend(def_line_nodes)
                        num_nodes += len(def_line_nodes)
                        # print "current : ------------------------" + str(current_node_json["children"])
                else:
                    current_node_json['children'].append(child_json)
                    # print "current : ------------------------" + str(current_node_json["children"])

            else:
               
                current_node_json['children'].append(child_json)
           

            queue_json.append(child_json)
            # print current_node_json
  
    return root_json, num_nodes   

def get_all_keys(dic):
    keys = []
    for key in dic:
        keys.append(key)
    return keys
def _traverse_tree(root):
    num_nodes = 1
    queue = [root]

    root_json = {
        "node": str(root.kind),

        "children": []
    }
    queue_json = [root_json]
    while queue:
       
        current_node = queue.pop(0)
        num_nodes += 1
        # print (_name(current_node))
        current_node_json = queue_json.pop(0)


        children = [x for x in current_node.child]
        queue.extend(children)
        
        for child in children:
       
            #print child.kind

            child_json = {
                "node": str(child.kind),
                "children": []
            }

            current_node_json['children'].append(child_json)


            queue_json.append(child_json)
            # print current_node_json
  
    return root_json, num_nodes

def parse_to_training_trees(infile,outfile_no_dependency,outfile_def_use,outfile_def_into_use,outfile_use_into_def):
    parse_pickle_to_training_trees_with_dependency(infile,outfile_no_dependency,outfile_def_use,outfile_def_into_use,outfile_use_into_def)
    # if with_dependency == "1":
       
        # parse_pickle_to_training_trees_with_dependency(infile,"./data/java_algorithms_tree.pkl","./data/java_algorithms_tree_dependency.pkl")
    # else:
       
        # parse_pickle_to_training_trees(infile,outfile)

        
if __name__ == "__main__": 
    # example: parse_pickle_to_training_trees("algorithms.pkl","algorithm_trees.pkl",True)
    parse_to_training_trees(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
    # with open("./pb_slice_pkl/bfs/cpp/1274.c.slice.pkl", 'rb') as file_handler:
    #     tree = pickle.load(file_handler)

    # line_dict, use_dict = _traverse_tree_to_extract_dependencies(tree.element)
    # sample, size = _traverse_tree_to_extract_context(tree.element,line_dict,use_dict)
    # print sample
