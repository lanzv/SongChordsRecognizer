// Select TemplateVoter model and display its upload form
function toTemplateVoter(e) {
    // Change button styles
    document.getElementById("predictors").style.background = "#efefef";
    document.getElementById("predictors").style.color = "#7d7d7d";
    document.getElementById("template_voter").style.background = "#010056";
    document.getElementById("template_voter").style.color = "#d0d0d0";
    // Display the other form
    document.getElementById("predictors-form").style.display = "none";
    document.getElementById("template_voter-form").style.display = "block";
}

// Select Predictors model and display its upload form
function toPredictors(e) {
    // Change button styles
    document.getElementById("template_voter").style.background = "#efefef";
    document.getElementById("template_voter").style.color = "#7d7d7d";
    document.getElementById("predictors").style.background = "#010056";
    document.getElementById("predictors").style.color = "#d0d0d0";
    // Display the other form
    document.getElementById("predictors-form").style.display = "block";
    document.getElementById("template_voter-form").style.display = "none";
}
