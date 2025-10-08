document.addEventListener("DOMContentLoaded", () => {

    const $root_elem = document.querySelector("#interest-graph-ui");
    const $map = $root_elem.querySelector(".interest-graph-ui-map");
    const $input = $root_elem.querySelector(".interest-graph-ui-input");
    const $result = $root_elem.querySelector(".interest-graph-ui-result");
    const $feature_select = $root_elem.querySelector(".interest-graph-ui-select");
    let graph_data = null;

    function on_data_loaded(data) {
        $result.innerText = "Data Loaded!";
        graph_data = data;
        window.graph_data = graph_data;

        graph_data.vertex_map = {};
        for (const i in graph_data.vertices) {
            graph_data.vertex_map[graph_data.vertices[i]] = {
                index: i,
                count: graph_data.vertex_counts[i],
                edges: [],
            };
        }
        graph_data.edges = graph_data.edges.map(
            ([a, b, count]) => [graph_data.vertices[a], graph_data.vertices[b], count]
        );
        for (let [a, b, count] of graph_data.edges) {
            graph_data.vertex_map[a].edges.push([b, count]);
            graph_data.vertex_map[b].edges.push([a, count]);
        }

        $feature_select.innerHTML = [{name: "edges"}].concat(graph_data.vertex_positions).map(i => {
            return `<option value="${i.name}">${i.name}</option>`;
        });
        $feature_select.onchange = () => query_word($input.value);

        if ($input.value) {
            query_word($input.value);
        }
    }

    function get_edges(vertices) {
        const vert_set = new Set(vertices);
        const x = graph_data.edges.filter(
            ([a, b, count]) => vert_set.has(a) & vert_set.has(b)
        );
        return x;
    }

    function get_edge_suggestions(word) {
        const v = graph_data.vertex_map[word];
        if (!v) { return []; }
        return v.edges;
    }

    function calc_distance(pos1, pos2) {
        const dx = pos1[0] - pos2[0];
        const dy = pos1[1] - pos2[1];
        return Math.sqrt(dx*dx + dy*dy);
    }

    function get_closest_vertices(word, feature_name) {
        const v = graph_data.vertex_map[word];
        if (!v) { return []; }
        const positions = graph_data.vertex_positions.filter(i => i.name === feature_name)[0].data;
        const word_pos = positions[graph_data.vertex_map[word].index];
        const distances = positions.map((pos, i) => [calc_distance(pos, word_pos), i]);
        distances.sort((a, b) => a[0] <= b[0] ? -1 : 1);
        return distances.map(d => [graph_data.vertices[d[1]], Math.round(d[0]*100)/100]);
    }

    function render_map(word, vertices) {
        let min_x = null, max_x = null, min_y = null, max_y = null;
        const items = [];
        for (const vert of vertices) {
            const label = vert[0];
            const vertex = graph_data.vertex_map[label];
            const pos = graph_data.vertex_positions[vertex.index];
            items.push({label, x: pos[0], y: pos[1]});
            if (min_x === null || pos[0] < min_x) min_x = pos[0];
            if (min_y === null || pos[1] < min_y) min_y = pos[1];
            if (max_x === null || pos[0] > max_x) max_x = pos[0];
            if (max_y === null || pos[1] > max_y) max_y = pos[1];
        }
        const rect = $map.getBoundingClientRect();

        const pad = .1;
        let html = "";
        for (const item of items) {
            const x = ((item.x - min_x) / (max_x - min_x) * (1. - pad) + pad/2) * rect.width;
            const y = ((item.y - min_y) / (max_y - min_y) * (1. - pad) + pad/2) * rect.height;
            html += `<div class="interest-graph-ui-map-item interest-graph-word" data-word="${item.label}"`
                + ` style="position: absolute; top: ${y}px; left: ${x}px">${item.label}</div>`
        }
        $map.innerHTML = html;
        hook_words();
    }

    let network = null;
    function update_network(vertices) {
        if (!network) {
            network = new vis.Network(
                $map,
                {},
                {
                    autoResize: true,
                    width: "100%",
                    height: "400px",
                    layout: {
                        improvedLayout: true,
                        randomSeed: 23,
                    },
                    nodes: {
                        shape: "dot",
                        font: {
                            color: "#ccc",
                            size: 40,
                        }
                        //widthConstraint: {maximum: 25},
                        //heightConstraint: {maximum: 25},
                    },
                    edges: {
                        arrows: "to",
                        arrowStrikethrough: false,
                        scaling: {
                            min: 2,
                            max: 10,
                        }
                    },
                    physics: {
                        barnesHut: {
                            //springLength: 200,
                        },
                        stabilization: {
                            iterations: 300,
                        }
                    }
                },
            );
            //network.on("click", on_network_click);
            //network.on("stabilizationProgress", e => status_msg(
            //    `stabilizing network ${e.iterations}/${e.total}`
            //));
            //network.on("stabilizationIterationsDone", e => status_msg());
            window.network = network;
        }
        //console.log({e: get_edges(vertices.map(v => v[0])), v: vertices.map(v => v[0])});

        let distances = vertices.map(e => e[1]);
        const max_dist = distances.reduce((d, a) => d + a, 0);
        distances = distances.map(d => 1. - d / max_dist);
        vertices = vertices.map((e, i) => [...e, distances[i]]);

        const vis_nodes = [];
        const vis_edges = [];
        for (const vert of vertices) {
            const label = vert[0];
            const vertex = graph_data.vertex_map[label];
            const pos = graph_data.vertex_positions[vertex.index];
            vis_nodes.push({
                id: label,
                label: label,
                x: pos[0],
                y: pos[1],
                color: {
                    background: "#ccc",
                    border: "#aaa",
                    highlight: "#eee",
                },
                value: graph_data.vertex_map[label].count,
            });
            if (vert !== vertices[0]) {
                vis_edges.push({
                    from: vertices[0][0],
                    to: label,
                    value: vert[2],
                    //label: other.weight,
                    //color: TYPE_COLOR_MAPPING[entry.type][0],
                });
            }
        }
        /*
        for (const edge of get_edges(vertices.map(v => v[0]))) {
            vis_edges.push({
                from: edge[0],
                to: edge[1],
            });
        }
         */

        network.setData({
            nodes: new vis.DataSet(vis_nodes),
            edges: new vis.DataSet(vis_edges),
        });
    }

    function count_links(word1, word2) {
        const v = graph_data.vertex_map[word1];
        if (!v) return 0;
        for (const [w2, count] of v.edges) {
            if (w2 === word2) return count
        }
        return 0;
    }

    function percent(x, n) {
        const p = Math.round(x / n * 10000) / 100;
        return `${p}%`;
    }

    function render_result_table(word, vertices) {
        let html = (
            `<table class="interest-graph-ui-table"><thead><tr>`
            + `<th>interest</th>`
            + `<th>used</th>`
            + `<th>used with ${word}</th>`
            + `<th>distance to ${word}</th>`
            + `</tr></thead><tbody>`
        );
        for (const vert of vertices) {
            const count = graph_data.vertex_map[vert[0]].count;
            html += (
                `<tr class="interest-graph-word" data-word="${vert[0]}">`
                + `<td >${vert[0]}</td>`
                + `<td class="align-right mono">${count} (${percent(count, graph_data.num_users)})</td>`
                + `<td class="align-right mono">${count_links(word, vert[0])}</td>`
                + `<td class="align-right mono">${vert[1]}</td>`
                + `</tr>`
            );
        }
        html += '</tbody></table>';
        $result.innerHTML = html;
        hook_words();
    }

    function query_word(word) {
        $input.value = word;
        const feature_name = $feature_select.value;
        const vertices = feature_name === "edges"
            ? get_edge_suggestions(feature_name)
            : get_closest_vertices(word, feature_name).slice(0, 100);
        render_result_table(word, vertices);
        render_map(word, vertices);
        //update_network(vertices);
    }

    function hook_words() {
        $root_elem.querySelectorAll('[data-word]').forEach($elem => {
            $elem.onclick = () => query_word($elem.getAttribute("data-word"));
        });
    }

    function hook_ui() {

        $input.onchange = () => query_word($input.value);

        fetch("/posts/2025/interest-graph.json")
            .then(r => r.json())
            .catch(e => $result.innerText = "Sorry, could not load the data.")
            .then(on_data_loaded)

    }


    hook_ui();

});