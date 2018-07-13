<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Rxivist: Popular biology pre-print papers ranked</title>

    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css" integrity="sha384-WskhaSGFgHYWDcbwN70/dfYBj47jz9qbsMId/iRN3ewGhXQFZCSftd1LZCfmhktB" crossorigin="anonymous">
    <link href="https://fonts.googleapis.com/css?family=Open+Sans:700" rel="stylesheet">
    <link rel="stylesheet" href="/static/rxivist.css">
  </head>

  <body>
  <br>
    <div class="container" id="main">
      <div class="row" id="header">
        <div class="col col-sm-10">
          <a href="/"><img src="/static/rxivist_logo_bad.png"></a>
          <div><em>The most popular articles on bioRxiv</em></div>
        </div>
      </div>
      <div class="row">
        <div class="col">
          <h2>Articles by {{data["given"]}} {{data["surname"]}}</h2>
          <ul>
            % for result in data["articles"]:
              <li><strong>{{result["title"]}}</strong>
                <ul>
                  <li>
                    % for i, author in enumerate(result["authors"]):
                      <a href="/authors/{{author["id"]}}">{{author["name"]}}</a>{{", " if i < (len(result["authors"]) - 1) else ""}}
                    % end
                  </li>
                  <li>All-time downloads rank: <strong>{{result["ranks"]["alltime"]}}</strong> out of {{result["ranks"]["out_of"]}}</li>
                  <li>Year-to-date downloads rank: <strong>{{result["ranks"]["ytd"]}}</strong> out of {{result["ranks"]["out_of"]}}</li>
                </ul>
              </li>
            % end
          </div>
        </div>
      </div>

      <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js" integrity="sha384-ZMP7rVo3mIykV+2+9J3UJ46jBk0WLaUAdn689aCwoqbBJiSnjAK/l8WvCWPIPm49" crossorigin="anonymous"></script>
      <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/js/bootstrap.min.js" integrity="sha384-smHYKdLADwkXOn1EmN1qk/HfnUcbVRZyYmZ4qpPea6sjB/pTJ0euyQp0Mk8ck+5T" crossorigin="anonymous"></script>
    </div>
    <div class="container">
      <div class="row">
        <div id="footer" class="col-sm-12">
          <p class="pull-right"><a href="http://blekhmanlab.org/">Blekhman<span class="footer-altcolor">Lab</span></a>
        </div>
      </div>
    </div>
  </body>
</html>