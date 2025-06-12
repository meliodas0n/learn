
function Header(props) {
  const title = props.title
  return <div>{title ? title : 'Default Title'}</div>
}

function HomePage() {
  const names = ['Ada Lovelace', 'Grace Hopper', 'Margaret Hamilton']

  const [likes, setLikes] = React.useState(0)
  function handleClick() {
    console.log("Increment by 1")
    setLikes(value + 1)
  }

  return (
    <div>
      <Header title="Develop. Preview. Ship." />
      <ul>
        {names.map(name => (
          <li>{name}</li>
        ))}
      </ul>
      <button onClick={handleClick}>Like ({likes})</button>
    </div>
  )
}