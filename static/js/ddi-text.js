const input = document.getElementById('input');
const loader = document.getElementById('loader');
const textInput = document.getElementById('textInput');
const entitiesBtn = document.getElementById('entitiesBtn');
const relationsBtn = document.getElementById('relationsBtn');
const outputEntities = document.getElementById('outputEntities');
const outputRelations = document.getElementById('outputRelations');

entitiesBtn.addEventListener('click', () => {
    const text = textInput.value;
    if (text !== '') {
        const json = { 'text': text };
        loader.classList.remove('d-none');
        fetch('http://localhost:5000/api/ner', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(json),
        }).then(response => response.json()).then(data => {
            input.classList.add('d-none');
            outputEntities.classList.remove('d-none');
            loader.classList.add('d-none');
            populateEntitiesTable(data);
        }).catch((error) => console.error('Error:', error));
    }
});

relationsBtn.addEventListener('click', () => {
    const text = textInput.value;
    if (text !== '') {
        const json = { 'text': text };
        loader.classList.remove('d-none');
        fetch('http://localhost:5000/api/re', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(json),
        }).then(response => response.json()).then(data => {
            input.classList.add('d-none');
            outputRelations.classList.remove('d-none');
            loader.classList.add('d-none');
            populateRelationsTable(data);
        }).catch((error) => console.error('Error:', error));
    }
});

function populateRelationsTable(data) {
    const table = document.getElementById('relationsTableBody');
    data.forEach(e => {
        let row = table.insertRow();
        let drugOne = row.insertCell(0);
        drugOne.innerHTML = e.first;
        let drugTwo = row.insertCell(1);
        drugTwo.innerHTML = e.second;
        let ddi = row.insertCell(2);
        ddi.innerHTML = e.ddi;
    });
    document.getElementById('backBtn2').addEventListener('click', backToInputText);
}

function populateEntitiesTable(data) {
    const table = document.getElementById('entitiesTableBody');
    for (const property in data) {
        const drugList = data[property];
        for (const drug in drugList) {
            let row = table.insertRow();
            let type = row.insertCell(0);
            type.innerHTML = property;
            let name = row.insertCell(1);
            name.innerHTML = drugList[drug].name;
        }
    }
    document.getElementById('backBtn1').addEventListener('click', backToInputText);
}

function backToInputText() {
    input.classList.remove('d-none');
    outputEntities.classList.add('d-none');
    outputRelations.classList.add('d-none');
}