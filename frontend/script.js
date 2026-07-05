const API = "http://127.0.0.1:8000";

async function enrollPerson(){

    const formData = new FormData();

    formData.append(
        "name",
        document.getElementById("name").value
    );

    formData.append(
        "image",
        document.getElementById("enrollImage").files[0]
    );

    const response = await fetch(
        API + "/enroll",
        {
            method:"POST",
            body:formData
        }
    );

    const data = await response.json();

    alert(data.message || "Enrolled Successfully");
}

async function recognizePerson(){

    const formData = new FormData();

    formData.append(
        "image",
        document.getElementById("recognizeImage").files[0]
    );

    const response = await fetch(
        API + "/recognize",
        {
            method:"POST",
            body:formData
        }
    );

    const data = await response.json();

    if(data.success){

        document.getElementById("result").innerHTML=
        `
        <h2>${data.name}</h2>
        <p>Confidence : ${data.confidence}%</p>
        `;

    }else{

        document.getElementById("result").innerHTML=
        `
        <h2>Unknown Person</h2>
        <p>Confidence : ${data.confidence}%</p>
        `;
    }

}