#include <vector>
#include <stdio.h>
#include "../maxflow-v3.03.src/graph.h"
#include <climits>

namespace maxflow {
    typedef Graph<float,float,float> GraphType;

    class Maxflow {
        public:
            // if add_nodes_auto is TRUE, then the graph is filled with nodes from 0..estimated_num_of_nodes-1 
            Maxflow(int estimated_num_of_nodes=0, int estimated_num_of_edges=0, bool add_nodes_auto=true);
            ~Maxflow();
            void add_nodes(int num);
            void add_tweights(int node, float source_w, float sink_w);
            void add_multiple_tweights(std::vector<int> nodes, std::vector<float> source_ws, std::vector<float> sink_ws);
            void add_edge(int node_i, int node_j, float capacity, float rev_capacity);
            void add_multiple_edges(std::vector<int> node_is, std::vector<int> node_js, std::vector<float> capacities, std::vector<float> rev_capacities);

            // the nodes which ended in sink are in sink_nodes vector
            std::vector<int> get_sink_nodes();

            float maxflow();

        private:
            GraphType *g;
        };
}