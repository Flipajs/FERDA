// (from maxflow README)
//#################################################################
//
// 3. Example usage.
//
// This section shows how to use the library to compute
// a minimum cut on the following graph:
//
//                 SOURCE
//                /       \
//              1/         \2
//              /      3    \
//            node0 -----> node1
//              |   <-----   |
//              |      4     |
//              \            /
//              5\          /6
//                \        /
//                   SINK
//
///////////////////////////////////////////////////

#include <stdio.h>
#include <climits>
#include "maxflow.h"
#include <vector>
#include <iostream>

using namespace maxflow;

int main()
{
    Maxflow *mf = new Maxflow(5, 40, true);
    mf->add_tweights(0, 0, 0);
    mf->add_tweights(1, 0, 0);
    mf->add_tweights(2, INT_MAX, 0);
    mf->add_tweights(3, 0, 1);
    mf->add_tweights(4, 0, 1);

    mf->add_edge(3, 0, 1, 1);
    mf->add_edge(0, 1, 1, 1);
    mf->add_edge(1, 4, 1, 1);
    mf->add_edge(4, 2, 1, 1);

    mf->maxflow();
    std::vector<int> sink = mf->get_sink_nodes();

    std::cout << sink.size();
    // printf("there are %d nodes in sink", sink.size());

    // typedef Graph<int,int,int> GraphType;
    // GraphType *g = new GraphType(/*estimated # of nodes*/ 2, /*estimated # of edges*/ 1);
        
    // int n_nodes = 5;
    // for (int i = 0; i < n_nodes; i++) {
    //     g->add_node();
    // }   

    // // g->add_tweights(0, 0, INT_MAX);
    // // g->add_tweights(1, 0, 0);
    // // g->add_tweights(2, 0, 1);

    // // g->add_edge(0, 2, 0, 0);
    // // // g->add_edge(2, 0, 0, 0);
    // // g->add_edge(2, 1, 0, 1);
    // // // g->add_edge(1, 2, 1, 1);


    // // g->add_tweights(0, 0, INT_MAX);
    // g->add_tweights(0, 0, 0);
    // g->add_tweights(1, 0, 0);
    // g->add_tweights(2, INT_MAX, 0);
    // g->add_tweights(3, 0, 1);
    // g->add_tweights(4, 0, 1);
    
    // // g->add_edge(0, 4, 0, 0);
    // g->add_edge(3, 0, 1, 1);
    // g->add_edge(0, 1, 1, 1);
    // g->add_edge(1, 4, 1, 1);
    // g->add_edge(4, 2, 1, 1);

    // // g->add_edge(4, 0, 0, INT_MAX);
    // // g->add_edge(1, 4, 10, 10);
    // // g->add_edge(2, 1, 10, 10);
    // // g->add_edge(5, 2, 10, 10);
    // // g->add_edge(3, 5, 10, 10);

    
    // int flow = g -> maxflow();
    
    // printf("Flow = %d\n", flow);
    // printf("Minimum cut:\n");
    // for (int i = 0; i < n_nodes; i++) {
    //     if (g->what_segment(i) == GraphType::SOURCE)
    //         printf("node%d is in the SOURCE set\n", i);
    //     else
    //         printf("node%d is in the SINK set\n", i);
    // }
    
    // delete g;
    
    return 0;
}