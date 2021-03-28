// Select TemplateVoter model and display its upload form
function toTemplateVoter(e) {
    // Change button styles
    document.getElementById("statistical_model").style.background = "#efefef";
    document.getElementById("statistical_model").style.color = "#7d7d7d";
    document.getElementById("template_voter").style.background = "#010056";
    document.getElementById("template_voter").style.color = "#d0d0d0";
    // Display the other form
    document.getElementById("statistical_model-form").style.display = "none";
    document.getElementById("template_voter-form").style.display = "block";
}

// Select StatisticalModel model and display its upload form
function toStatisticalModel(e) {
    // Change button styles
    document.getElementById("template_voter").style.background = "#efefef";
    document.getElementById("template_voter").style.color = "#7d7d7d";
    document.getElementById("statistical_model").style.background = "#010056";
    document.getElementById("statistical_model").style.color = "#d0d0d0";
    // Display the other form
    document.getElementById("statistical_model-form").style.display = "block";
    document.getElementById("template_voter-form").style.display = "none";
}
