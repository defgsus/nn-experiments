let mutator_data = null;

function renderImages() {
    if (!mutator_data)
        return;

    const w = mutator_data.resolution[1],
          h = mutator_data.resolution[0];

    var html = ''
    for (const row of mutator_data.images) {
        html += `<div class="image-row">`;
        html += row.map(image =>
            `<div class="image" data-x="${image.x}" data-y="${image.y}">
                <img src="${image.filename}" width="${w * 2}">
            </div>`
        ).join("");
        html += '</div>';
    }

    document.querySelector("#images").innerHTML = html;
    document.querySelectorAll("#images .image").forEach($image => {
        $image.onclick = () => {
            mutate(
                parseInt($image.getAttribute("data-x")),
                parseInt($image.getAttribute("data-y"))
            );
        }
    });
}

function updateImages() {
    if (!mutator_data)
        return;

    const w = mutator_data.resolution[1],
          h = mutator_data.resolution[0];

    for (const y in mutator_data.images) {
        for (const x in mutator_data.images[y]) {
            const $div = document.querySelector(`#images .image[data-x="${x}"][data-y="${y}"]`);
            if ($div) {
                $div.querySelector("img").setAttribute("src", mutator_data.images[y][x].filename);
            }
        }
    }
}

async function fetchData() {
    const response = await fetch("mutator/");
    mutator_data = await response.json();
    renderImages();
}

async function mutator(action, data) {
    const response = await fetch("mutator/", {
        method: "POST",
        body: JSON.stringify({
            "action": action,
            ...data,
        })
    });
    mutator_data = await response.json();
    updateImages();
}

async function mutate(x, y) {
    mutator("mutate", {
        "x": x,
        "y": y,
        "amount": parseFloat(document.querySelector("#mutation-amount").value),
    });
}

async function undo() {
    mutator("undo");
}

document.addEventListener("DOMContentLoaded", function() {
    fetchData();
    document.querySelector("#undo").onclick = () => undo();
});
