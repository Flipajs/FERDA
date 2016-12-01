#include "maxflow.h"

using namespace maxflow;

Maxflow::Maxflow(int estimated_num_of_nodes, int estimated_num_of_edges, bool add_nodes_auto) {
    g = new GraphType(estimated_num_of_nodes, estimated_num_of_edges);

    if (add_nodes_auto) {
        g->add_node(estimated_num_of_nodes);
    }
}

Maxflow::~Maxflow() {
    delete g;
}

void Maxflow::add_nodes(int num) {
    g->add_node(num);
}

void Maxflow::add_tweights(int node, float source_w, float sink_w) {
    g->add_tweights(node, source_w, sink_w);
}

void Maxflow::add_multiple_tweights(std::vector<int> nodes, std::vector<float> source_ws, std::vector<float> sink_ws) {
    for (int i = 0; i < nodes.size(); i++) {
        g->add_tweights(nodes[i], source_ws[i], sink_ws[i]);
    }
}

void Maxflow::add_edge(int node_i, int node_j, float capacity, float rev_capacity) {
    g->add_edge(node_i, node_j, capacity, rev_capacity);
}

void Maxflow::add_multiple_edges(std::vector<int> node_is, std::vector<int> node_js, std::vector<float> capacities, std::vector<float> rev_capacities) {
    for (int i = 0; i < node_is.size(); i++) {
        g->add_edge(node_is[i], node_js[i], capacities[i], rev_capacities[i]);
    }
}

float Maxflow::maxflow() {
    return g->maxflow();
}    

std::vector<int> Maxflow::get_sink_nodes() {
    std::vector<int> sink_nodes;

    for (int i = 0; i < g->get_node_num(); i++) {
        if (g->what_segment(i) == GraphType::SINK)
            sink_nodes.push_back(i);
    }

    return sink_nodes;
}




