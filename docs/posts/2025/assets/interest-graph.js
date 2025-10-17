document.addEventListener("DOMContentLoaded", () => {

    const $root_elem = document.querySelector("#interest-graph-ui");
    const $info = $root_elem.querySelector(".interest-graph-ui-info");
    const $input = $root_elem.querySelector(".interest-graph-ui-input");
    const $result = $root_elem.querySelector(".interest-graph-ui-result");
    const $result_num = $root_elem.querySelector(".interest-graph-ui-num");
    const $feature_select = $root_elem.querySelector(".interest-graph-ui-select");
    const $map = $root_elem.querySelector(".interest-graph-ui-map");
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

        $info.innerText = (
            `profiles: ${graph_data.num_users}`
            + `\ninterests: ${graph_data.num_vertices}`
            + `\ninterest connections: ${graph_data.num_edges}`
        );
        $result_num.innerHTML = [10, 50, 100, 500, "all"].map(i => {
            return `<option value="${i}">${i === 'all' ? `all (${graph_data.num_vertices})` : i}</option>`;
        });
        $result_num.onchange = () => {
            query_word($input.value);
            set_url_hash($input.value, $result_num.value, $feature_select.value);
        }
        $feature_select.innerHTML = [{name: "edge count"}].concat(graph_data.vertex_positions).map(i => {
            return `<option value="${i.name.replaceAll(' ', '_')}">${i.name}</option>`;
        });
        $feature_select.value = "pca1000-tsne2";
        $feature_select.onchange = () => {
            query_word($input.value);
            set_url_hash($input.value, $result_num.value, $feature_select.value);
        }

        if (!update_ui_from_url_hash() && $input.value) {
            query_word($input.value);
        }
    }

    function get_vertices_edges(vertices) {
        const vert_set = new Set(vertices);
        const x = graph_data.edges.filter(
            ([a, b, count]) => vert_set.has(a) & vert_set.has(b)
        );
        return x;
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

    function count_links(word1, word2) {
        const v = graph_data.vertex_map[word1];
        if (!v) return 0;
        for (const [w2, count] of v.edges) {
            if (w2 === word2) return count
        }
        return 0;
    }

    function get_closest_vertices_by_edge_count(word) {
        const v = graph_data.vertex_map[word];
        if (!v) { return []; }
        return [[word, 0]].concat(v.edges);
    }

    function percent(x, n) {
        const p = Math.round(x / n * 10000) / 100;
        return `${p}%`;
    }

    function render_result_table(word, vertices, count, with_distance) {
        if (!vertices || !vertices.length) {
            $result.innerText = `Sorry, '${word}' is not in the dataset.`;
            return;
        }
        const sum_links = graph_data.vertex_map[word].edges.reduce((a, e) => a + e[1], 0);
        let html = (
            `<table class="interest-graph-ui-table"><thead><tr>`
            + `<th>interest</th>`
            + `<th>times used</th>`
            + `<th style="max-width: 8rem;">connections with '${word}'<br/>(${sum_links} at all)</th>`
            + (with_distance ? `<th style="max-width: 8rem;">distance to '${word}'</th>` : ``)
            + `</tr></thead><tbody>`
        );
        for (const vert of vertices.slice(0, count)) {
            const count = graph_data.vertex_map[vert[0]].count;
            const num_links = count_links(word, vert[0]);
            html += (
                `<tr class="interest-graph-word" data-word="${vert[0]}">`
                + `<td >${vert[0]}</td>`
                + `<td class="align-right mono">${count} (${percent(count, graph_data.num_users)})</td>`
                + `<td class="align-right mono">` + (num_links ? `${num_links} (${percent(num_links, sum_links)})` : '') + `</td>`
                + (with_distance ? `<td class="align-right mono">${vert[1] ? vert[1] : ''}</td>` : ``)
                + `</tr>`
            );
        }

        html += '</tbody></table>';
        $result.innerHTML = html;
        hook_words();

        if (window.nnblog_hook_tables) {
            window.nnblog_hook_tables();
        }
    }

    function render_map(word, vertices, feature_name) {
        let html = "";
        if (feature_name !== "edge_count") {
            let min_x = null, max_x = null, min_y = null, max_y = null;
            const items = [];
            for (const vert of vertices.slice(0, 50)) {
                const label = vert[0];
                const vertex = graph_data.vertex_map[label];
                const positions = graph_data.vertex_positions.filter(i => i.name === feature_name)[0].data;
                const pos = positions[vertex.index];
                items.push({label, x: pos[0], y: pos[1], dist: vert[1]});
                if (min_x === null || pos[0] < min_x) min_x = pos[0];
                if (min_y === null || pos[1] < min_y) min_y = pos[1];
                if (max_x === null || pos[0] > max_x) max_x = pos[0];
                if (max_y === null || pos[1] > max_y) max_y = pos[1];
            }
            for (const vert of vertices.slice(30)) {
                const label = vert[0];
                const vertex = graph_data.vertex_map[label];
                const positions = graph_data.vertex_positions.filter(i => i.name === feature_name)[0].data;
                const pos = positions[vertex.index];
                if (pos[0] >= min_x && pos[0] <= max_x && pos[1] >= min_y && pos[1] <= max_y) {
                    items.push({label, x: pos[0], y: pos[1], dist: vert[1]});
                }
            }
            const rect = $map.getBoundingClientRect();
            const pad = .1;
            for (const item of items) {
                const x = ((item.x - min_x) / (max_x - min_x) * (1. - pad) + pad/2) * rect.width;
                const y = rect.height - 1 - ((item.y - min_y) / (max_y - min_y) * (1. - pad) + pad/2) * rect.height;
                let title = item.label;
                if (item.label !== word) {
                    title = `${item.label}\ndistance to '${word}': ${item.dist}`;
                }
                html += (
                    `<div class="interest-graph-ui-map-item interest-graph-word ${item.label === word ? 'highlight' : ''}" data-word="${item.label}"`
                    + ` style="position: absolute; top: ${y}px; left: ${x}px" title="${title}">${item.label}</div>`
                );
            }
        }
        $map.innerHTML = html;
        hook_words();
    }

    function escape_html(str){
        var p = document.createElement("p");
        p.appendChild(document.createTextNode(str));
        return p.innerHTML;
    }

    function query_word(word) {
        word = escape_html(word);
        $input.value = word;
        const feature_name = $feature_select.value;
        const vertices = feature_name === "edge_count"
            ? get_closest_vertices_by_edge_count(word)
            : get_closest_vertices(word, feature_name);

        const count = $result_num.value === "all"
            ? graph_data.num_vertices
            : parseInt($result_num.value);

        if (feature_name !== "edge_count") {
            $map.removeAttribute("hidden");
        } else {
            $map.setAttribute("hidden", "hidden");
        }

        render_result_table(word, vertices, count, feature_name !== "edge_count");
        render_map(word, vertices, feature_name);
    }

    function set_url_hash(word, count, feature_name) {
        //window.history.pushState({}, "", window.location);
        if (count === graph_data.num_vertices)
            count = "all";
        const param = new URLSearchParams({w: word, n: count, f: feature_name});
        window.location.hash = `#${param.toString()}`;
    }

    function update_ui_from_url_hash() {
        const p = new URLSearchParams(window.location.hash.slice(1));
        let update = false;
        if (p.get("w")) { $input.value = p.get("w"); update = true; }
        if (p.get("n")) { $result_num.value = p.get("n"); update = true; }
        if (p.get("f")) { $feature_select.value = p.get("f"); update = true; }
        if (update && $input.value) { query_word($input.value); return true; }
    }

    function scroll_to_ui() {
        var rect = $root_elem.getBoundingClientRect();
        if (!(
            rect.top >= 0 && rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        )) {
            $root_elem.scrollIntoView();
        }
    }

    function hook_words() {
        document.querySelectorAll('[data-word]').forEach($elem => {
            $elem.onclick = (e) => {
                e.stopPropagation();
                e.preventDefault();
                scroll_to_ui();
                query_word($elem.getAttribute("data-word"));
                set_url_hash($input.value, $result_num.value, $feature_select.value);
            }
        });
    }

    function hook_links() {
        document.querySelectorAll("a").forEach($elem => {
            if ($elem.getAttribute("href")?.indexOf("=") > 0) {
                $elem.onclick = (e) => {
                    scroll_to_ui();
                }
            }

        });
    };

    function hook_ui() {
        hook_links();

        $input.onchange = () => {
            query_word($input.value);
            set_url_hash($input.value, $result_num.value, $feature_select.value);
        }
        window.addEventListener("hashchange", e => {
            update_ui_from_url_hash();
        })

        $info.innerHTML = "Loading data ...";
        $root_elem.removeAttribute("hidden");
        fetch("/posts/2025/interest-graph.json")
            .then(r => r.json())
            .catch(e => {
                $result.innerText = "Sorry, could not load the data.";
            })
            .then(data => { if (!data) throw Error("The XHR requests was probably blocked."); return data; })
            .then(on_data_loaded)
            .catch(e => {
                $result.innerText = `Sorry, something went wrong: ${e.message}`;
            });
    }

    hook_ui();

});