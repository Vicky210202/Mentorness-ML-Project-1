async function makePrediction() {
    const first_name = document.getElementById('first_name').value;
    const last_name = document.getElementById('last_name').value;
    const sex = document.getElementById('sex').value;
    const doj = document.getElementById('doj').value;
    const current_date = document.getElementById('current_date').value;
    const designation = document.getElementById('designation').value;
    const age = parseInt(document.getElementById('age').value);
    const unit = document.getElementById('unit').value;
    const leaves_used = parseInt(document.getElementById('leaves_used').value);
    const leaves_remaining = parseInt(document.getElementById('leaves_remaining').value);
    const ratings = parseInt(document.getElementById('ratings').value);
    const past_exp = parseInt(document.getElementById('past_exp').value);

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            first_name: first_name,
            last_name: last_name,
            sex: sex,
            doj: doj,
            current_date: current_date,
            designation: designation,
            age: age,
            unit: unit,
            leaves_used: leaves_used,
            leaves_remaining: leaves_remaining,
            ratings: ratings,
            past_exp: past_exp
        }),
    });

    const data = await response.json();
    document.getElementById('result').innerText = `Hello! ${first_name} ${last_name}, your predicted salary is: $ ${data.prediction}`;
}
